from __future__ import annotations
import json
import subprocess
from pathlib import Path
import yt_dlp
from src.settings.settings import settings
from src.utils.paths import reel_folder
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class Scraper:
    def __init__(self, whisper_model: str | None = None):
        self.whisper_model_name = whisper_model or settings.whisper_model
        self._whisper_model = None
        self.device_used = "cpu"
        self._check_ffmpeg()


    @staticmethod
    def _check_ffmpeg():
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(
                "ffmpeg not found on PATH. Install it first so streams can merge."
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
        if self._whisper_model is None:
            import torch
            import whisper

            if torch.cuda.is_available():
                try:
                    logger.info("🚀 NVIDIA GPU detected. Initializing CUDA acceleration...")
                    self._whisper_model = whisper.load_model(self.whisper_model_name, device="cuda")
                    self.device_used = "cuda"
                    logger.info("✅ Whisper loaded onto GPU.")
                except Exception as gpu_err:
                    logger.warning(f"⚠️ GPU init failed ({gpu_err}). Falling back to CPU...")
                    self._whisper_model = whisper.load_model(self.whisper_model_name, device="cpu")
                    self.device_used = "cpu"
            else:
                logger.info("ℹ️ CUDA unavailable. Running Whisper on CPU...")
                self._whisper_model = whisper.load_model(self.whisper_model_name, device="cpu")
                self.device_used = "cpu"

        return self._whisper_model


    def _transcribe(self, media_path: Path) -> str:
        model = self._load_whisper()
        use_fp16 = self.device_used == "cuda"
        logger.info(f"🎙️ Transcribing [{media_path.name}] on [{self.device_used.upper()}]...")
        result = model.transcribe(str(media_path), fp16=use_fp16, verbose=True)
        return result["text"].strip()
    

    def _get_transcript(self, url: str, folder: Path, video_path: Path, has_audio: bool) -> str | None:
        """Transcribe from the video, or via an audio-only fallback download."""
        
        if has_audio:
            try:
                return self._transcribe(video_path)
            except Exception as e:
                logger.error(f"❌ Video transcription failed: {e}")
                return None

        logger.info("🔇 No audio in video. Trying audio-only fallback...")
        audio_tmpl = str(folder / "audio_only.%(ext)s")
        audio_opts = {
            "outtmpl": audio_tmpl,
            "format": "bestaudio/best",
            "quiet": True,
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                ydl.download([url])
            audio_files = list(folder.glob("audio_only.*"))
            if not audio_files:
                logger.info("🔇 Silent reel — no audio track available.")
                return None
            transcript = self._transcribe(audio_files[0])
            return transcript
        except Exception as e:
            logger.error(f"❌ Audio fallback failed: {e}")
            return None
        finally:
            for tmp in folder.glob("audio_only.*"):
                tmp.unlink(missing_ok=True)


    def scrape(self, url: str) -> dict:
        """Return the reel schema dict, downloading/transcribing only if needed."""

        logger.info(f"🔎 Resolving reel info: {url}")
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)

        reel_id = info.get("id") or "reel"
        title = info.get("title") or reel_id
        folder = reel_folder(title, fallback=reel_id)
        info_path = folder / settings.info_name

        if info_path.exists():
            logger.info(f"📦 Folder already exists ({folder.name}). Returning stored result.")
            with open(info_path, "r", encoding="utf-8") as f:
                return json.load(f)

        folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"📥 Downloading media into {folder}...")
        ydl_opts = {
            "outtmpl": str(folder / "video.%(ext)s"),
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "writeinfojson": False,
            "quiet": False,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        video_path = folder / settings.video_name
        if not video_path.exists():
            candidates = [p for p in folder.glob("video.*") if p.suffix in (".mp4", ".mkv", ".webm")]
            video_path = candidates[0] if candidates else None

        caption = info.get("description") or ""

        result: dict = {
            "url": url,
            "title": title,
            "caption": caption,
            "video_path": str(video_path) if video_path else None,
            "has_audio": False,
            "transcript": None,
        }

        if video_path and video_path.exists():
            result["has_audio"] = self._has_audio_stream(video_path)
            transcript = self._get_transcript(url, folder, video_path, result["has_audio"])
            if transcript is not None:
                result["transcript"] = transcript
                with open(folder / settings.transcript_name, "w", encoding="utf-8") as f:
                    f.write(transcript)

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info("✅ Reel scrape complete.")
        return result
