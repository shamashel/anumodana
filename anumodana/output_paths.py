from __future__ import annotations

from pathlib import Path


DEFAULT_AUDIO_EXTENSION = ".mp3"
DEFAULT_MANIFEST_NAME = "_anumodana_review_manifest.csv"


def audio_output_path(source_path: Path) -> Path:
    if source_path.suffix.lower() == DEFAULT_AUDIO_EXTENSION:
        return source_path
    return source_path.with_suffix(DEFAULT_AUDIO_EXTENSION)


def cleaned_vtt_output_path(source_path: Path) -> Path:
    return source_path.with_suffix(".vtt")


def correction_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.qwen.vtt")


def raw_vtt_output_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}.parakeet.raw.vtt")


def review_json_output_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}.review.json")


def review_md_output_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}.review.md")


def resolve_manifest_path(root: Path, manifest_arg: str) -> Path:
    if manifest_arg:
        return Path(manifest_arg).expanduser().resolve()
    return root / DEFAULT_MANIFEST_NAME
