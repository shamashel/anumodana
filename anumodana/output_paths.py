from __future__ import annotations

from pathlib import Path


DEFAULT_AUDIO_EXTENSION = ".mp3"
DEFAULT_MANIFEST_NAME = "_anumodana_review_manifest.csv"
DEFAULT_COLLECTION_NAME = "Ajahn Wade Recordings"
DEFAULT_REVISION_DIR_NAME = "Transcript Revision"


def audio_output_path(source_path: Path) -> Path:
    if source_path.suffix.lower() == DEFAULT_AUDIO_EXTENSION:
        return source_path
    return source_path.with_suffix(DEFAULT_AUDIO_EXTENSION)


def transcript_output_path(source_path: Path) -> Path:
    return source_path.with_suffix(".txt")


def correction_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.qwen.vtt")


def revision_root_path(root: Path) -> Path:
    return root.parent / DEFAULT_REVISION_DIR_NAME


def _revision_output_path(root: Path, source_path: Path, filename: str) -> Path:
    relative_parent = source_path.relative_to(root).parent
    return revision_root_path(root) / relative_parent / filename


def cleaned_vtt_output_path(root_or_source: Path, source_path: Path | None = None) -> Path:
    if source_path is None:
        return root_or_source.with_suffix(".vtt")
    return _revision_output_path(root_or_source, source_path, f"{source_path.stem}.vtt")


def raw_vtt_output_path(root_or_source: Path, source_path: Path | None = None) -> Path:
    if source_path is None:
        return root_or_source.with_name(f"{root_or_source.stem}.parakeet.raw.vtt")
    return _revision_output_path(root_or_source, source_path, f"{source_path.stem}.parakeet.raw.vtt")


def review_json_output_path(root_or_source: Path, source_path: Path | None = None) -> Path:
    if source_path is None:
        return root_or_source.with_name(f"{root_or_source.stem}.review.json")
    return _revision_output_path(root_or_source, source_path, f"{source_path.stem}.review.json")


def review_md_output_path(root_or_source: Path, source_path: Path | None = None) -> Path:
    if source_path is None:
        return root_or_source.with_name(f"{root_or_source.stem}.review.md")
    return _revision_output_path(root_or_source, source_path, f"{source_path.stem}.review.md")


def resolve_manifest_path(root: Path, manifest_arg: str) -> Path:
    if manifest_arg:
        return Path(manifest_arg).expanduser().resolve()
    return revision_root_path(root) / DEFAULT_MANIFEST_NAME
