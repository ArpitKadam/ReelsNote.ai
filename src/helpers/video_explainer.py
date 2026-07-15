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
    "Format:\n"
    "- Output ONLY valid Markdown. No preamble, no sign-off, no conversational "
    "text.\n"
    "- Use `##`/`###` headings, bullet lists, numbered steps, `**bold**` for key "
    "terms, and fenced code blocks for any code or commands.\n"
    "- Structure it logically: a short overview, then the detailed teaching, then "
    "key takeaways."
)


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


def describe(state: dict) -> str | None:
    """Return the educational explanation for the reel in `state`.

    Reads `video_path` from state, writes/loads `video_explanation.md` next to
    the video. Returns the Markdown string, or None on failure.
    """

    video_path_str = state.get("video_path")
    if not video_path_str:
        logger.error("No video_path in state; cannot generate explanation.")
        return None

    video_path = Path(video_path_str)
    explanation_path = video_path.parent / settings.explanation_name

    if explanation_path.exists():
        logger.info("📦 video_explanation.md exists. Returning stored explanation.")
        return explanation_path.read_text(encoding="utf-8")

    try:
        b64 = _encode_video_to_base64(video_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return None

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
        return None
    except Exception as e:
        logger.error(f"Unexpected error during video description: {e}")
        return None

    explanation = response.choices[0].message.content
    if explanation:
        explanation_path.write_text(explanation, encoding="utf-8")
        logger.info(f"✅ Explanation saved to {explanation_path.name}")
    return explanation
