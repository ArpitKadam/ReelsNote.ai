from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    output_dir: Path = Path("output")

    whisper_model: str = "small"
    # Force Whisper's language instead of auto-detecting from the first 30s
    # (auto-detect misfires to Khmer/Nynorsk/etc. on music or noisy intros and
    # then hallucinates gibberish). Set to "" / "auto" to re-enable detection.
    whisper_language: str = os.getenv("WHISPER_LANGUAGE", "en")

    info_name: str = "info.json"
    transcript_name: str = "transcript.txt"
    video_name: str = "video.mp4"
    explanation_name: str = "video_explanation.md"
    report_name: str = "report.md"
    question_name: str = "question.txt"
    reel_pdf_name: str = "reel.pdf"

    notes_pdf_path: Path = Path("notes.pdf")
    manifest_name: str = "notes_manifest.json"

    # Instagram auth for yt-dlp: path to Netscape cookies.txt, or browser name
    # ("firefox", "edge", "chrome") to read cookies directly from it.
    ig_cookies_file: str = os.getenv("IG_COOKIES_FILE", "")
    ig_cookies_browser: str = os.getenv("IG_COOKIES_BROWSER", "")

    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    nvidia_model: str = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
    nvidia_report_model: str = os.getenv("NVIDIA_REPORT_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # Videos larger than this (raw MB) are split into segments before sending,
    # since inline base64 of the whole file exceeds NVIDIA's payload limit.
    nvidia_max_video_mb: float = float(os.getenv("NVIDIA_MAX_VIDEO_MB", "20"))
    # Approximate target raw size per segment when splitting.
    nvidia_chunk_mb: float = float(os.getenv("NVIDIA_CHUNK_MB", "12"))


settings = Settings()
