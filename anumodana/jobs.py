from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anumodana.output_paths import (
    audio_output_path,
    cleaned_vtt_output_path,
    raw_vtt_output_path,
    transcript_output_path,
    review_json_output_path,
    review_md_output_path,
)


SUPPORTED_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".wav",
    ".webm",
}

SOURCE_PRIORITY = {
    ".mp4": 0,
    ".mkv": 1,
    ".mov": 2,
    ".webm": 3,
    ".m4v": 4,
    ".mpeg": 5,
    ".mpg": 6,
    ".m4a": 7,
    ".mp3": 8,
    ".flac": 9,
    ".aac": 10,
    ".wav": 11,
}

EXCLUDED_DIR_NAMES = {".git", ".venv", ".uv-cache", ".uv-python", "__pycache__", "Transcript Revision"}
TRANSIENT_SIDECAR_SUFFIXES = (
    ".distil.whisper.json",
    ".whisper.json",
    ".whisper-input.wav",
)


@dataclass(frozen=True)
class Job:
    source_path: Path
    audio_path: Path
    transcript_path: Path
    raw_vtt_path: Path
    cleaned_vtt_path: Path
    review_json_path: Path
    review_md_path: Path
    needs_audio: bool
    needs_transcript: bool
    needs_raw_vtt: bool
    needs_cleaned_vtt: bool
    needs_review: bool


def choose_source(candidates: list[Path]) -> Path:
    return min(
        candidates,
        key=lambda path: (
            SOURCE_PRIORITY.get(path.suffix.lower(), 99),
            path.suffix.lower(),
            path.name.lower(),
        ),
    )


def _group_sources(root: Path) -> dict[tuple[Path, str], list[Path]]:
    grouped: dict[tuple[Path, str], list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if path.name.endswith(".whisper-input.wav"):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        grouped.setdefault((path.parent, path.stem), []).append(path)
    return grouped


def iter_preferred_sources(root: Path) -> list[Path]:
    sources = [choose_source(candidates) for candidates in _group_sources(root).values()]
    sources.sort(key=lambda path: str(path).lower())
    return sources


def discover_jobs(root: Path, overwrite: bool, run_review: bool) -> tuple[list[Job], int]:
    jobs: list[Job] = []
    skipped = 0

    for source_path in iter_preferred_sources(root):
        audio_path = audio_output_path(source_path)
        transcript_path = transcript_output_path(source_path)
        raw_vtt_path = raw_vtt_output_path(root, source_path)
        cleaned_vtt_path = cleaned_vtt_output_path(root, source_path)
        review_json_path = review_json_output_path(root, source_path)
        review_md_path = review_md_output_path(root, source_path)
        has_audio = audio_path.exists()
        has_transcript = transcript_path.exists()
        has_raw_vtt = raw_vtt_path.exists()
        has_cleaned_vtt = cleaned_vtt_path.exists()
        has_review = review_json_path.exists() and review_md_path.exists()

        if (
            not overwrite
            and has_audio
            and has_transcript
            and has_raw_vtt
            and has_cleaned_vtt
            and (has_review or not run_review)
        ):
            skipped += 1
            continue

        jobs.append(
            Job(
                source_path=source_path,
                audio_path=audio_path,
                transcript_path=transcript_path,
                raw_vtt_path=raw_vtt_path,
                cleaned_vtt_path=cleaned_vtt_path,
                review_json_path=review_json_path,
                review_md_path=review_md_path,
                needs_audio=overwrite or not has_audio,
                needs_transcript=overwrite or not has_transcript,
                needs_raw_vtt=overwrite or not has_raw_vtt,
                needs_cleaned_vtt=overwrite or not has_cleaned_vtt,
                needs_review=run_review and (overwrite or not has_review),
            )
        )

    return jobs, skipped


def cleanup_transient_artifacts(job: Job) -> list[Path]:
    keep_paths = {
        job.source_path.resolve(),
        job.audio_path.resolve(),
        job.transcript_path.resolve(),
        job.raw_vtt_path.resolve(),
        job.cleaned_vtt_path.resolve(),
        job.review_json_path.resolve(),
        job.review_md_path.resolve(),
    }
    candidates: list[Path] = []
    stem_path = job.source_path.with_suffix("")
    for suffix in TRANSIENT_SIDECAR_SUFFIXES:
        candidates.append(Path(f"{stem_path}{suffix}"))
    if job.source_path.suffix.lower() != ".wav":
        candidates.append(job.source_path.with_suffix(".wav"))

    removed: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue
        if resolved in keep_paths or not candidate.exists() or not candidate.is_file():
            continue
        candidate.unlink()
        removed.append(candidate)
    return removed
