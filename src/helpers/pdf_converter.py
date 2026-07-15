"""Convert a reel's `report.md` into a page in the global `notes.pdf`.

Pipeline role: runs after the `report` node. For each reel it builds a small
per-reel PDF whose FIRST page is the question the reel answers and whose
remaining pages are the rendered report. Every reel ever processed is tracked in
a manifest; on each run the whole `notes.pdf` is rebuilt as:

    [ index pages ] + [ reel #1: question + content ] + [ reel #2 ... ] + ...

The index is built as its own PDF (it can span multiple pages) and then simply
concatenated in front of the reel PDFs — exactly the "make a separate pdf for
the index and join afterwards" approach.

Math written as LaTeX between dollar signs — e.g. ``$Gini = 1 - \\sum (p_k^2)$``
or ``$$...$$`` — is rendered to images with matplotlib's mathtext engine so it
displays as real notation instead of raw source.
"""

from __future__ import annotations
import io
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt
import markdown as md_lib
import pymupdf
from src.settings.settings import settings
from src.utils.logging_config import get_logger
from src.utils.paths import reel_folder

logger = get_logger(__name__)

_PAGE = pymupdf.paper_rect("a4")
_MARGIN = 48  # points
_CONTENT_RECT = _PAGE + (_MARGIN, _MARGIN, -_MARGIN, -_MARGIN)

_BASE_CSS = """
* { font-family: sans-serif; }
body { font-size: 11px; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 22px; margin: 0 0 12px 0; color: #0b3d91; }
h2 { font-size: 17px; margin: 18px 0 8px 0; color: #0b3d91;
     border-bottom: 1px solid #d0d7de; padding-bottom: 3px; }
h3 { font-size: 14px; margin: 14px 0 6px 0; color: #244; }
h4 { font-size: 12px; margin: 12px 0 4px 0; }
p  { margin: 6px 0; }
ul, ol { margin: 6px 0 6px 0; padding-left: 20px; }
li { margin: 3px 0; }
strong { color: #111; }
code { font-family: monospace; background: #f2f4f7; padding: 1px 3px;
       font-size: 10px; }
pre { background: #f6f8fa; padding: 8px; font-size: 10px; line-height: 1.35;
      white-space: pre-wrap; word-wrap: break-word; }
pre code { background: none; padding: 0; }
blockquote { color: #555; border-left: 3px solid #d0d7de; margin: 6px 0;
             padding: 2px 10px; }
table { border-collapse: collapse; margin: 8px 0; }
th, td { border: 1px solid #d0d7de; padding: 4px 8px; font-size: 10px; }
th { background: #f2f4f7; }
img.math-inline { vertical-align: middle; }
img.math-block { display: block; margin: 8px auto; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 12px 0; }
"""

_QUESTION_CSS = """
* { font-family: sans-serif; }
body { color: #1a1a1a; }
.kicker { font-size: 12px; letter-spacing: 3px; color: #0b3d91;
          text-transform: uppercase; margin-bottom: 24px; }
.question { font-size: 26px; line-height: 1.4; font-weight: bold;
            color: #111; }
.meta { margin-top: 28px; font-size: 11px; color: #667085; }
"""

_INDEX_CSS = """
* { font-family: sans-serif; }
body { color: #1a1a1a; font-size: 12px; }
h1 { font-size: 26px; color: #0b3d91; margin: 0 0 4px 0; }
.sub { color: #667085; font-size: 11px; margin-bottom: 20px; }
.row { padding: 7px 0; border-bottom: 1px solid #eaecef; }
.num { color: #0b3d91; font-weight: bold; }
.q { }
.pg { color: #667085; float: right; }
"""

_FENCE_RE = re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]+`)", re.DOTALL)
_BLOCK_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$", re.DOTALL)


def _render_math_png(expr: str, block: bool) -> bytes | None:
    """Render a single LaTeX expression to transparent PNG bytes, or None."""

    expr = expr.strip()
    if not expr:
        return None
    fontsize = 20 if block else 15
    fig = plt.figure(figsize=(0.01, 0.01))
    try:
        # Opaque white background + black text: no alpha channel, so no
        # soft-mask that PDF viewers can misrender as a solid black box.
        fig.text(0, 0, f"${expr}$", fontsize=fontsize, color="black")
        buf = io.BytesIO()
        fig.savefig(
            buf,
            dpi=200,
            format="png",
            bbox_inches="tight",
            pad_inches=0.05,
            transparent=False,
            facecolor="white",
        )
        # Flatten to opaque RGB (drop alpha) so no PDF viewer can render the
        # image's soft-mask as a solid black box.
        from PIL import Image

        img = Image.open(io.BytesIO(buf.getvalue()))
        if img.mode != "RGB":
            bg = Image.new("RGB", img.size, "white")
            bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = bg
        out = io.BytesIO()
        img.save(out, format="png")
        return out.getvalue()
    except Exception as e:  # unsupported LaTeX -> caller keeps raw text
        logger.warning(f"Could not render math '{expr[:40]}': {e}")
        return None
    finally:
        plt.close(fig)


def _substitute_math(text: str, archive: pymupdf.Archive, counter: list[int]) -> str:
    """Replace math spans in `text` with <img> tags backed by `archive`."""

    def _emit(expr: str, block: bool) -> str:
        png = _render_math_png(expr, block)
        if png is None:
            return f"$${expr}$$" if block else f"${expr}$"  # fall back to raw
        idx = counter[0]
        counter[0] += 1
        name = f"math_{idx}.png"
        archive.add(png, name)
        # Scale by rendered pixel height so inline math matches the text size.
        try:
            h_px = pymupdf.Pixmap(png).height
        except Exception:
            h_px = 40
        if block:
            height_em = min(3.0, max(1.4, h_px / 200 * 2.2))
            return f'<p style="text-align:center"><img class="math-block" src="{name}" style="height:{height_em:.2f}em"></p>'
        height_em = min(2.2, max(1.0, h_px / 200 * 2.0))
        return f'<img class="math-inline" src="{name}" style="height:{height_em:.2f}em">'

    text = _BLOCK_MATH_RE.sub(lambda m: _emit(m.group(1), block=True), text)
    text = _INLINE_MATH_RE.sub(lambda m: _emit(m.group(1), block=False), text)
    return text


def _markdown_to_html(md_text: str, archive: pymupdf.Archive) -> str:
    """Markdown (with LaTeX math) -> HTML body string, images added to archive.

    Code spans/blocks are protected so `$` inside them is left untouched.
    """

    counter = [0]
    parts = _FENCE_RE.split(md_text)
    # Odd indices are the protected code segments; even indices are prose.
    for i in range(0, len(parts), 2):
        parts[i] = _substitute_math(parts[i], archive, counter)
    processed = "".join(parts)

    html_body = md_lib.markdown(
        processed,
        extensions=["fenced_code", "tables", "sane_lists"],
    )
    # Unwrap `<pre><code ...>` to a bare `<pre>`: pymupdf's Story renders a
    # `<code>` child inside `<pre>` with a hardcoded dark theme (unreadable
    # black boxes), while a bare `<pre>` honours our light CSS. Inline `<code>`
    # spans are left as-is (they render fine).
    html_body = re.sub(r"<pre>\s*<code[^>]*>", "<pre>", html_body)
    html_body = html_body.replace("</code></pre>", "</pre>")
    return f"<html><head></head><body>{html_body}</body></html>"


def _story_to_pdf_bytes(html: str, css: str, archive: pymupdf.Archive | None) -> bytes:
    """Paginate an HTML story into a standalone PDF, returned as bytes."""

    story = pymupdf.Story(html=html, user_css=css, archive=archive)
    buf = io.BytesIO()
    writer = pymupdf.DocumentWriter(buf)
    more = True
    while more:
        dev = writer.begin_page(_PAGE)
        more, _ = story.place(_CONTENT_RECT)
        story.draw(dev)
        writer.end_page()
    writer.close()
    return buf.getvalue()


def _pdf_page_count(pdf_bytes: bytes) -> int:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count


def _esc(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _question_page_pdf(question: str, title: str, url: str) -> bytes:
    """One-page PDF: the question this reel answers (the reel's cover page)."""

    html = (
        "<html><body>"
        '<div class="kicker">Question</div>'
        f'<div class="question">{_esc(question)}</div>'
        f'<div class="meta">{_esc(title)}<br>{_esc(url)}</div>'
        "</body></html>"
    )
    return _story_to_pdf_bytes(html, _QUESTION_CSS, None)


def _build_reel_pdf(question: str, title: str, url: str, report_md: str) -> bytes:
    """Per-reel PDF: page 1 = question, remaining pages = rendered report."""

    archive = pymupdf.Archive()
    content_html = _markdown_to_html(report_md, archive)

    q_pdf = _question_page_pdf(question, title, url)
    content_pdf = _story_to_pdf_bytes(content_html, _BASE_CSS, archive)

    out = pymupdf.open()
    with pymupdf.open(stream=q_pdf, filetype="pdf") as q:
        out.insert_pdf(q)
    with pymupdf.open(stream=content_pdf, filetype="pdf") as c:
        out.insert_pdf(c)
    data = out.tobytes()
    out.close()
    return data


def _build_index_pdf(entries: list[dict], start_pages: list[int]) -> bytes:
    """Index PDF listing every reel's question and the page it begins on."""

    rows = []
    for i, (entry, pg) in enumerate(zip(entries, start_pages), start=1):
        q = _esc(entry.get("question") or entry.get("title") or "Untitled")
        pg_txt = f"p.{pg}" if pg else ""
        rows.append(
            f'<div class="row"><span class="num">{i}.</span> '
            f'<span class="pg">{pg_txt}</span>'
            f'<span class="q"> {q}</span></div>'
        )
    html = (
        "<html><body>"
        "<h1>Reel Notes — Index</h1>"
        f'<div class="sub">{len(entries)} reel(s)</div>'
        + "".join(rows)
        + "</body></html>"
    )
    return _story_to_pdf_bytes(html, _INDEX_CSS, None)


def _manifest_path() -> Path:
    return settings.output_dir / settings.manifest_name


def _load_manifest() -> list[dict]:
    path = _manifest_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("reels", []) if isinstance(data, dict) else []
    except Exception as e:
        logger.warning(f"Could not read manifest ({e}); starting fresh.")
        return []


def _save_manifest(reels: list[dict]) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"reels": reels}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _upsert(reels: list[dict], entry: dict) -> list[dict]:
    """Add the reel, or update it in place if its URL is already present."""

    for i, r in enumerate(reels):
        if r.get("url") and r.get("url") == entry.get("url"):
            reels[i] = entry
            return reels
    reels.append(entry)
    return reels


def convert(state: dict) -> str | None:
    """Add this reel to the global `notes.pdf` and rebuild it. Returns its path.

    Steps:
      1. Build this reel's PDF (question page + report) and cache it.
      2. Record it in the manifest (dedup by URL).
      3. Rebuild every reel PDF that is missing from disk.
      4. Build the index, compute page offsets, and concatenate:
         index + all reel PDFs  ->  notes.pdf
    """

    report_md = state.get("report")
    if not report_md:
        video_path_str = state.get("video_path")
        folder = (
            Path(video_path_str).parent
            if video_path_str
            else reel_folder(state.get("title", "reel"))
        )
        report_file = folder / settings.report_name
        if report_file.exists():
            report_md = report_file.read_text(encoding="utf-8")
    if not report_md:
        logger.error("No report available; skipping PDF conversion.")
        return None

    video_path_str = state.get("video_path")
    folder = (
        Path(video_path_str).parent
        if video_path_str
        else reel_folder(state.get("title", "reel"))
    )
    folder.mkdir(parents=True, exist_ok=True)

    title = state.get("title") or "Reel Report"
    url = state.get("url") or ""
    question = state.get("question") or title  # fall back to title if absent

    logger.info("Rendering reel PDF (question page + report)...")
    reel_pdf_bytes = _build_reel_pdf(question, title, url, report_md)
    reel_pdf_path = folder / settings.reel_pdf_name
    reel_pdf_path.write_bytes(reel_pdf_bytes)

    reels = _upsert(
        _load_manifest(),
        {
            "url": url,
            "title": title,
            "question": question,
            "reel_pdf": str(reel_pdf_path),
        },
    )
    _save_manifest(reels)

    page_counts: list[int] = []
    valid: list[dict] = []
    for entry in reels:
        p = Path(entry.get("reel_pdf", ""))
        if not p.exists():
            logger.warning(f"Missing reel PDF for '{entry.get('title')}'; skipping in notes.")
            continue
        page_counts.append(_pdf_page_count(p.read_bytes()))
        valid.append(entry)

    if not valid:
        logger.error("No reel PDFs available; cannot build notes.pdf.")
        return None

    index_pages = 1
    start_pages: list[int] = []
    for _ in range(4):
        offset = index_pages + 1
        start_pages = []
        for pc in page_counts:
            start_pages.append(offset)
            offset += pc
        index_pdf_bytes = _build_index_pdf(valid, start_pages)
        new_index_pages = _pdf_page_count(index_pdf_bytes)
        if new_index_pages == index_pages:
            break
        index_pages = new_index_pages

    notes = pymupdf.open()
    with pymupdf.open(stream=index_pdf_bytes, filetype="pdf") as idx:
        notes.insert_pdf(idx)
    for entry in valid:
        with pymupdf.open(entry["reel_pdf"]) as reel:
            notes.insert_pdf(reel)

    notes_path = settings.notes_pdf_path
    notes.save(str(notes_path))
    notes.close()
    logger.info(f"notes.pdf rebuilt with {len(valid)} reel(s) -> {notes_path}")
    return str(notes_path)
