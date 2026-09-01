from __future__ import annotations
import re
from pathlib import Path
from src.settings.settings import settings

_ILLEGAL = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_WS = re.compile(r"\s+")
_MAX_LEN = 120


def sanitize_title(title: str, fallback: str = "reel") -> str:
    """Turn an arbitrary reel title into a safe folder name.

    Strips illegal characters, collapses whitespace, trims trailing dots/spaces
    (Windows rejects those), and caps length. Falls back if nothing usable
    remains.
    """

    cleaned = _ILLEGAL.sub("", title or "")
    cleaned = _WS.sub(" ", cleaned).strip()
    cleaned = cleaned.strip(". ")
    cleaned = cleaned[:_MAX_LEN].strip(". ")
    return cleaned or fallback


def reel_folder(title: str, reel_id: str = "", fallback: str = "reel") -> Path:
    """Folder for a reel's artifacts: output/{safe_title} [{reel_id}]/.

    The id suffix keeps folders unique — many reels share a generic title
    like "Video by <creator>", which would otherwise collide and serve one
    reel's cache for another.
    """

    name = sanitize_title(title, fallback)
    if reel_id and reel_id not in name:
        name = f"{name} [{reel_id}]"
    return settings.output_dir / name
