from __future__ import annotations
import base64
from pathlib import Path
from openai import OpenAI, APIError
from src.settings.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

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

    size_mb = video_path.stat().st_size / (1024 * 1024)
    if size_mb > 20:
        logger.warning(
            f"⚠️ Video is {size_mb:.1f}MB; base64 grows it to ~{size_mb * 1.33:.1f}MB, "
            "which may exceed NVIDIA API payload limits."
        )
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _client() -> OpenAI:
    return OpenAI(base_url=settings.nvidia_base_url, api_key=settings.nvidia_api_key)


def describe(state: dict) -> tuple[str | None, str | None]:
    """Return `(explanation, question)` for the reel in `state`.

    Reads `video_path` from state, writes/loads `video_explanation.md` and
    `question.txt` next to the video. `explanation` is the Markdown string and
    `question` is the single question the reel answers (used as the reel's
    title/index entry). Either may be None on failure.
    """

    video_path_str = state.get("video_path")
    if not video_path_str:
        logger.error("No video_path in state; cannot generate explanation.")
        return None, None

    video_path = Path(video_path_str)
    explanation_path = video_path.parent / settings.explanation_name
    question_path = video_path.parent / settings.question_name

    if explanation_path.exists():
        logger.info("📦 video_explanation.md exists. Returning stored explanation.")
        explanation = explanation_path.read_text(encoding="utf-8")
        question = (
            question_path.read_text(encoding="utf-8").strip()
            if question_path.exists()
            else None
        )
        return explanation, question

    try:
        b64 = _encode_video_to_base64(video_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return None, None

    logger.info("🚀 Sending video to NVIDIA NIM for educational analysis...")
    try:
        response = _client().chat.completions.create(
            model=settings.nvidia_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EDUCATIONAL_PROMPT},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=8192,
        )
    except APIError as e:
        logger.error(f"NVIDIA API error: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error during video description: {e}")
        return None, None

    raw = response.choices[0].message.content
    if not raw:
        return None, None

    explanation, question = _split_question(raw)
    if explanation:
        explanation_path.write_text(explanation, encoding="utf-8")
        logger.info(f"✅ Explanation saved to {explanation_path.name}")
    if question:
        question_path.write_text(question, encoding="utf-8")
        logger.info(f"❓ Question saved: {question}")
    return explanation, question
