from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_AUDIO_BITRATE = "48k"


def ensure_ffmpeg_on_path() -> str | None:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None

    ffmpeg_root = Path(local_app_data) / "Programs" / "ffmpeg"
    if not ffmpeg_root.exists():
        return None

    bin_directories = [path for path in ffmpeg_root.glob("*/bin") if path.is_dir()]
    if not bin_directories:
        return None

    newest_bin = max(bin_directories, key=lambda path: path.stat().st_mtime)
    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    if str(newest_bin) not in path_parts:
        os.environ["PATH"] = os.pathsep.join([str(newest_bin), *path_parts])

    return str(newest_bin)


def extract_audio_copy(source_path: Path, audio_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise FileNotFoundError(
            "FFmpeg was not found. Install an FFmpeg shared build or add ffmpeg.exe to PATH."
        )

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        DEFAULT_AUDIO_BITRATE,
        str(audio_path),
    ]
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stdout.strip() or f"ffmpeg exited with code {process.returncode}")
    if not audio_path.exists():
        raise RuntimeError("ffmpeg finished without creating the expected audio file.")


def get_media_duration_seconds(path: Path) -> float:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise FileNotFoundError(
            "FFprobe was not found. Install an FFmpeg shared build or add ffprobe.exe to PATH."
        )

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stdout.strip() or f"ffprobe exited with code {process.returncode}")

    try:
        return float(process.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe did not return a usable duration for {path}") from exc


def extract_chunk_wav(
    source_media: Path,
    chunk_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise FileNotFoundError(
            "FFmpeg was not found. Install an FFmpeg shared build or add ffmpeg.exe to PATH."
        )

    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(source_media),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(chunk_path),
    ]
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stdout.strip() or f"ffmpeg exited with code {process.returncode}")
    if not chunk_path.exists():
        raise RuntimeError("ffmpeg finished without creating the expected chunk WAV file.")
