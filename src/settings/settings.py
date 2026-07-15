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

    info_name: str = "info.json"
    transcript_name: str = "transcript.txt"
    video_name: str = "video.mp4"
    explanation_name: str = "video_explanation.md"
    report_name: str = "report.md"

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_report_model: str = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")

    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    nvidia_model: str = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"


settings = Settings()
