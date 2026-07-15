from __future__ import annotations
from pathlib import Path
from groq import Groq
from src.settings.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an expert technical writer and research analyst. You transform raw, "
    "messy source material from a short video (its caption, spoken transcript, and "
    "a model-generated visual explanation) into a polished, in-depth study report "
    "that stands on its own as a learning resource. You are thorough, precise, and "
    "never pad with fluff. You write in clean Markdown only."
)

REPORT_TEMPLATE = """Create a DETAILED study report from the following reel.

Source URL: {url}
Title: {title}

--- CAPTION ---
{caption}

--- TRANSCRIPT (spoken audio) ---
{transcript}

--- VIDEO EXPLANATION (visual/conceptual analysis) ---
{video_explanation}

=== YOUR TASK ===
Write a comprehensive, deeply informative Markdown report that teaches the
reader everything of value in this reel. Synthesize ALL three sources above into
a single coherent document — reconcile and merge overlapping information rather
than repeating it. Prefer depth over brevity: fully explain concepts, define
terms, and spell out reasoning and steps.

Use exactly this structure (omit a section only if there is genuinely no
material for it):

# {title}

## 1. Overview
A tight 3-5 sentence summary of what the reel is about and what the reader will
learn.

## 2. Key Concepts
A bulleted list of the core ideas, terms, or techniques — each with a one-line
definition.

## 3. Detailed Explanation
The heart of the report. Explain every important concept in depth, in a logical
order. Use `###` subheadings per topic. Explain the 'why' and 'how'. Include
formulas, code, or commands in fenced blocks where relevant.

## 4. Step-by-Step (if applicable)
If the reel demonstrates a process, give a clear numbered procedure the reader
could follow themselves.

## 5. Practical Takeaways
Actionable points, best practices, and common pitfalls to avoid.

## 6. Further Notes
Useful context, caveats, or related directions worth exploring.

Rules:
- Output ONLY the Markdown report. No preamble, no meta commentary, no sign-off.
- Do NOT invent facts unsupported by the sources; if something is unclear, note
  it briefly rather than fabricating.
- Be genuinely detailed — this should read like thorough study notes, not a
  caption rewrite."""

_MISSING = "(not available)"


def _client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


def generate(state: dict) -> str | None:
    """Return a detailed Markdown report for the reel in `state`.

    Writes/loads `report.md` in the reel's folder (inferred from `video_path`).
    Returns the Markdown string, or None on failure.
    """

    video_path_str = state.get("video_path")
    if video_path_str:
        folder = Path(video_path_str).parent
    else:
        from src.utils.paths import reel_folder
        folder = reel_folder(state.get("title", "reel"))

    report_path = folder / settings.report_name

    if report_path.exists():
        logger.info("📦 report.md exists. Returning stored report.")
        return report_path.read_text(encoding="utf-8")

    prompt = REPORT_TEMPLATE.format(
        url=state.get("url") or _MISSING,
        title=state.get("title") or "Reel Report",
        caption=state.get("caption") or _MISSING,
        transcript=state.get("transcript") or _MISSING,
        video_explanation=state.get("video_explanation") or _MISSING,
    )

    logger.info("🧠 Generating report with Groq...")
    try:
        response = _client().chat.completions.create(
            model=settings.groq_report_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=8192,
        )
    except Exception as e:
        logger.error(f"Groq API error during report generation: {e}")
        return None

    report = response.choices[0].message.content
    if report:
        folder.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        logger.info(f"✅ Report saved to {report_path.name}")
    return report
