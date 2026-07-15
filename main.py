from __future__ import annotations
import sys
from pathlib import Path
from rich import print as rprint

from src.pipeline.pipeline import build_graph
from src.utils.logging_config import get_logger

logger = get_logger("ReelsNote.ai")

DEFAULT_URL = "https://www.instagram.com/reel/DasfzZBxRTg/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA=="


def run(url: str) -> dict:
    graph = build_graph()
    if not Path("pipeline.png").exists():
        graph.get_graph().draw_mermaid_png(output_file_path="pipeline.png")

    final_state = graph.invoke({"url": url})

    rprint("\n[bold green]===== FINAL STATE =====[/bold green]")
    rprint(
        {
            "url": final_state.get("url"),
            "title": final_state.get("title"),
            "has_audio": final_state.get("has_audio"),
            "video_path": final_state.get("video_path"),
            "transcript_chars": len(final_state.get("transcript") or ""),
            "video_explanation_chars": len(final_state.get("video_explanation") or ""),
            "report_chars": len(final_state.get("report") or ""),
        }
    )
    return final_state


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    run(url)
