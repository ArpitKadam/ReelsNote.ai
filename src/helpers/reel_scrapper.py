"""
reel-scraper.py

Download public Instagram Reels (video + caption + metadata), with a
local Whisper fallback to generate a transcript if the downloaded video
ends up without an audio track.

No paid APIs used anywhere:
  - yt-dlp        -> downloading (free, open-source)
  - ffmpeg        -> muxing/probing (free, open-source, must be installed system-wide)
  - openai-whisper-> local transcription (free, runs on your machine)

Install:
    pip install yt-dlp openai-whisper
    # + ffmpeg must be on PATH (apt install ffmpeg / brew install ffmpeg / choco install ffmpeg)

Usage:
    python reel-scraper.py "https://www.instagram.com/reel/XXXXXXXXX/"

Output layout:
    output/{video_title}/
        video.mp4          <- the reel, with audio if available
        transcript.txt     <- only created if video has no audio track
        info.json          <- caption + metadata + status flags
"""

import json
import re
import subprocess
from pathlib import Path
import yt_dlp


class Scraper:

    def __init__(self, output_dir: str = "output", whisper_model: str = "small"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.whisper_model_name = whisper_model
        self._whisper_model = None
        self.device_used = "cpu"  # Default baseline tracker
        self._check_ffmpeg()

    @staticmethod
    def _sanitize(name: str) -> str:
        name = re.sub(r'[\\/*?:"<>|]', "", name)
        name = name.strip().rstrip(".")
        return name[:150] if name else "untitled_reel"

    @staticmethod
    def _check_ffmpeg():
        try:
            subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, check=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(
                "ffmpeg not found on PATH. Install it first to ensure streams can merge."
            )

    @staticmethod
    def _has_audio_stream(video_path: Path) -> bool:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True, text=True,
        )
        return bool(result.stdout.strip())
    
    def _load_whisper(self):
        """Loads Whisper onto GPU if available, with automatic CPU fallback."""
        if self._whisper_model is None:
            import torch
            import whisper

            # Check if PyTorch is correctly built with CUDA support for your RTX 3050
            if torch.cuda.is_available():
                try:
                    print("🚀 NVIDIA GPU Detected. Initializing CUDA acceleration...")
                    self._whisper_model = whisper.load_model(self.whisper_model_name, device="cuda")
                    self.device_used = "cuda"
                    print("✅ Whisper loaded successfully onto your RTX GPU!")
                except Exception as gpu_err:
                    print(f"⚠️ GPU init failed ({gpu_err}). Falling back to CPU...")
                    self._whisper_model = whisper.load_model(self.whisper_model_name, device="cpu")
                    self.device_used = "cpu"
            else:
                print("ℹ️ CUDA unavailable via PyTorch build. Running on CPU mode...")
                self._whisper_model = whisper.load_model(self.whisper_model_name, device="cpu")
                self.device_used = "cpu"

        return self._whisper_model
    
    def _transcribe(self, audio_or_video_path: Path) -> str:
        model = self._load_whisper()
        
        use_fp16 = True if self.device_used == "cuda" else False
        
        print(f"🎙️ Transcribing [{audio_or_video_path.name}] using [{self.device_used.upper()}]...")
        result = model.transcribe(str(audio_or_video_path), fp16=use_fp16, verbose=False)
        return result["text"].strip()
    
    def download(self, url: str) -> dict:
        with yt_dlp.YoutubeDL({"quiet": False, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)

        title = info.get("title") or info.get("id") or "reel"
        folder_name = self._sanitize(title)
        target_dir = self.output_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        video_template = str(target_dir / "video.%(ext)s")

        ydl_opts = {
            "outtmpl": video_template,
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "writeinfojson": True,
            "quiet": False,
            "noplaylist": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        video_path = target_dir / "video.mp4"
        if not video_path.exists():
            candidates = [f for f in target_dir.glob("video.*")
                        if f.suffix in (".mp4", ".mkv", ".webm")]
            video_path = candidates[0] if candidates else None

        info_json_path = target_dir / "video.info.json"
        if info_json_path.exists():
            with open(info_json_path, "r", encoding="utf-8") as f:
                raw_info = json.load(f)
        else:
            raw_info = info

        caption = raw_info.get("description", "") or ""

        result = {
            "url": url,
            "title": title,
            "caption": caption,
            "video_path": str(video_path) if video_path else None,
            "has_audio": False,
            "transcript": None,
        }

        if video_path and video_path.exists():
            has_audio = self._has_audio_stream(video_path)
            result["has_audio"] = has_audio

            if has_audio:
                try:
                    transcript = self._transcribe(video_path)
                    result["transcript"] = transcript
                    with open(target_dir / "transcript.txt", "w", encoding="utf-8") as f:
                        f.write(transcript)
                except Exception as e:
                    result["transcript_error"] = f"Video transcription failed: {e}"

            else:
                audio_template = str(target_dir / "audio_only.%(ext)s")
                audio_opts = {
                    "outtmpl": audio_template,
                    "format": "bestaudio/best",
                    "quiet": True,
                    "noplaylist": True,
                }
                try:
                    with yt_dlp.YoutubeDL(audio_opts) as ydl:
                        ydl.download([url])
                    audio_files = list(target_dir.glob("audio_only.*"))
                    if audio_files:
                        transcript = self._transcribe(audio_files[0])
                        result["transcript"] = transcript
                        with open(target_dir / "transcript.txt", "w", encoding="utf-8") as f:
                            f.write(transcript)
                    else:
                        result["transcript_error"] = "No audio track available (silent reel)."
                except Exception as e:
                    result["transcript_error"] = f"Audio download fallback failed: {e}"

        with open(target_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result





# Example Usage
# scraper = Scraper()
# output = scraper.download(url="")
# print(json.dumps(output, indent=2, ensure_ascii=False))