from __future__ import annotations
import base64
import math
import shutil
import subprocess
from pathlib import Path
from openai import OpenAI, APIError
from src.settings.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Shared, EXACT output contract reused by the single-shot and synthesis prompts
# so both paths produce identically-formatted results.
_OUTPUT_FORMAT = (
    "Output format — follow EXACTLY:\n"
    "1. The VERY FIRST line must be:\n"
    "   QUESTION: <one clear, specific question that this video answers>\n"
    "   Make it a natural question a learner would ask (e.g. "
    "'How does the Gini impurity measure node purity in a decision tree?'). "
    "One line only, no markdown, no quotes.\n"
    "2. Then a line containing only three dashes: ---\n"
    "3. Then the full educational explanation as valid Markdown:\n"
    "   - No preamble, no sign-off, no conversational text.\n"
    "   - Use `##`/`###` headings, bullet lists, numbered steps, `**bold**` for "
    "key terms, and fenced code blocks for any code or commands.\n"
    "   - Write any math as LaTeX between dollar signs, e.g. $Gini = 1 - \\sum "
    "(p_k^2)$ inline or $$...$$ on its own line.\n"
    "   - Structure it logically: a short overview, then the detailed teaching, "
    "then key takeaways."
)

EDUCATIONAL_PROMPT = (
    "You are a world-class educator and subject-matter expert. You are given a "
    "short social-media video. Your job is to extract and TEACH the underlying "
    "knowledge it contains — not to narrate what appears on screen.\n\n"
    "Produce a self-contained educational explanation that someone could learn "
    "from without ever watching the video.\n\n"
    "Rules:\n"
    "- Explain the CONCEPTS, mechanisms, techniques, or arguments presented. Go "
    "beyond the surface: state the 'why' and 'how', not just the 'what'.\n"
    "- Whenever a term, tool, formula, or process is mentioned, define it and "
    "explain it well enough that a motivated beginner understands it.\n"
    "- If the video demonstrates steps, reconstruct them as a clear, ordered "
    "procedure the reader could follow.\n"
    "- Add relevant context, caveats, or common mistakes that make the topic "
    "genuinely useful — but do not invent facts the video does not support.\n"
    "- Ignore filler, intros, background music, branding, and social-media "
    "calls-to-action (like/follow/subscribe).\n\n"
    + _OUTPUT_FORMAT
)

# Prompt for a SINGLE segment of a longer video. Output is raw study notes that
# are later merged — no QUESTION line, no strict format here.
_CHUNK_PROMPT_TEMPLATE = (
    "You are given ONE segment (segment {index} of {total}) of a longer "
    "educational social-media video. Extract every concept, definition, step, "
    "formula, and technique TAUGHT in THIS segment as detailed study notes. "
    "Teach the underlying knowledge — do not narrate what appears on screen. "
    "Ignore intros, branding, background music, and social-media "
    "calls-to-action.\n\n"
    "Output plain, detailed notes for this segment only. No preamble, no "
    "QUESTION line, no sign-off. It is fine if the segment starts or ends "
    "mid-topic; just capture what is taught."
)

# Prompt that merges per-segment notes into one coherent final explanation.
_SYNTHESIS_PROMPT_HEADER = (
    "You are a world-class educator. Below are study notes extracted from "
    "consecutive segments of a SINGLE short educational video, in order. Some "
    "topics may span segment boundaries or repeat. Merge them into ONE "
    "coherent, self-contained educational explanation that someone could learn "
    "from without ever watching the video. Deduplicate, order logically, "
    "define every term, and explain the 'why' and 'how' — but do not invent "
    "facts the notes do not support.\n\n"
    + _OUTPUT_FORMAT
    + "\n\nSEGMENT NOTES:\n"
)


def _split_question(raw: str) -> tuple[str, str | None]:
    """Split the model output into (markdown_explanation, question).

    Expects a leading `QUESTION: ...` line, then a `---` separator, then the
    Markdown body. Degrades gracefully if the model ignored the format.
    """

    question: str | None = None
    body = raw.strip()

    lines = body.splitlines()
    if lines and lines[0].strip().lower().startswith("question:"):
        question = lines[0].split(":", 1)[1].strip() or None
        rest = lines[1:]
        # Drop an immediately-following '---' separator line if present.
        while rest and rest[0].strip() in ("", "---"):
            rest.pop(0)
        body = "\n".join(rest).strip()

    return body, question


def _encode_video_to_base64(video_path: Path) -> str:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _client() -> OpenAI:
    return OpenAI(base_url=settings.nvidia_base_url, api_key=settings.nvidia_api_key)


def _call_nvidia(content: list | str, max_tokens: int = 8192) -> str | None:
    """Single NVIDIA NIM chat call. Returns the message text, or None on error."""

    try:
        response = _client().chat.completions.create(
            model=settings.nvidia_model,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
    except APIError as e:
        logger.error(f"NVIDIA API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during NVIDIA call: {e}")
        return None

    return response.choices[0].message.content


def _explain_video_file(video_path: Path, prompt: str) -> str | None:
    """Send one video file inline (base64) with `prompt`; return the raw reply."""

    b64 = _encode_video_to_base64(video_path)
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{b64}"},
        },
    ]
    return _call_nvidia(content)


def _video_duration_seconds(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _split_video(video_path: Path, chunk_dir: Path) -> list[Path]:
    """Split `video_path` into time-based segments each ~`nvidia_chunk_mb`.

    Uses stream-copy (`-c copy`), so cuts land on keyframes and segment sizes
    are approximate. Returns the ordered list of produced chunk files.
    """

    size_mb = video_path.stat().st_size / (1024 * 1024)
    num_chunks = max(2, math.ceil(size_mb / settings.nvidia_chunk_mb))

    duration = _video_duration_seconds(video_path)
    if duration <= 0:
        raise RuntimeError("Could not read video duration; cannot split for chunking.")

    segment_time = duration / num_chunks
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(chunk_dir / "chunk_%03d.mp4")

    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(video_path),
            "-c", "copy", "-map", "0",
            "-f", "segment",
            "-segment_time", f"{segment_time:.3f}",
            "-reset_timestamps", "1",
            pattern,
        ],
        check=True, capture_output=True,
    )

    return sorted(chunk_dir.glob("chunk_*.mp4"))


def _describe_chunked(video_path: Path) -> str | None:
    """Split an oversized video, explain each segment, synthesize one reply.

    Returns the final raw model output (QUESTION + --- + Markdown), or None.
    """

    chunk_dir = video_path.parent / "chunks"
    try:
        chunks = _split_video(video_path, chunk_dir)
        if not chunks:
            logger.error("Video splitting produced no chunks.")
            return None

        logger.info(f"🔪 Split video into {len(chunks)} segments for analysis.")

        notes: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            logger.info(f"Analyzing segment {i}/{len(chunks)}...")
            prompt = _CHUNK_PROMPT_TEMPLATE.format(index=i, total=len(chunks))
            reply = _explain_video_file(chunk, prompt)
            if reply and reply.strip():
                notes.append(f"### Segment {i}\n{reply.strip()}")
            else:
                logger.warning(f"Segment {i} returned no content; skipping.")

        if not notes:
            logger.error("All segments failed; no notes to synthesize.")
            return None

        logger.info("Synthesizing final explanation from segment notes...")
        synthesis_input = _SYNTHESIS_PROMPT_HEADER + "\n\n".join(notes)
        return _call_nvidia(synthesis_input)
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)


def describe(state: dict) -> tuple[str | None, str | None]:
    """Return `(explanation, question)` for the reel in `state`.

    Reads `video_path` from state, writes/loads `video_explanation.md` and
    `question.txt` next to the video. `explanation` is the Markdown string and
    `question` is the single question the reel answers (used as the reel's
    title/index entry). Either may be None on failure.

    Videos larger than `nvidia_max_video_mb` are split into segments, explained
    individually, and merged — inline base64 of the whole file would otherwise
    exceed NVIDIA API payload limits and fail with a connection error.
    """

    video_path_str = state.get("video_path")
    if not video_path_str:
        logger.error("No video_path in state; cannot generate explanation.")
        return None, None

    video_path = Path(video_path_str)
    explanation_path = video_path.parent / settings.explanation_name
    question_path = video_path.parent / settings.question_name

    if explanation_path.exists():
        logger.info("video_explanation.md exists. Returning stored explanation.")
        explanation = explanation_path.read_text(encoding="utf-8")
        question = (
            question_path.read_text(encoding="utf-8").strip()
            if question_path.exists()
            else None
        )
        return explanation, question

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return None, None

    size_mb = video_path.stat().st_size / (1024 * 1024)

    if size_mb > settings.nvidia_max_video_mb:
        logger.warning(
            f"Video is {size_mb:.1f}MB (base64 ~{size_mb * 1.33:.1f}MB), over the "
            f"{settings.nvidia_max_video_mb:.0f}MB inline limit; splitting into segments."
        )
        raw = _describe_chunked(video_path)
    else:
        logger.info("Sending video to NVIDIA NIM for educational analysis...")
        raw = _explain_video_file(video_path, EDUCATIONAL_PROMPT)

    if not raw:
        return None, None

    explanation, question = _split_question(raw)
    if explanation:
        explanation_path.write_text(explanation, encoding="utf-8")
        logger.info(f"Explanation saved to {explanation_path.name}")
    if question:
        question_path.write_text(question, encoding="utf-8")
        logger.info(f"Question saved: {question}")
    return explanation, question
