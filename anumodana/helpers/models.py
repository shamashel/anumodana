"""Pydantic models for structured data throughout the pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Fixer (correction) response models
# ---------------------------------------------------------------------------
class CorrectionItem(BaseModel):
    id: int = Field(description="The unique cue ID from the original VTT.")
    text: str = Field(description="The cleaned text for this cue.")


class CorrectionResponse(BaseModel):
    items: list[CorrectionItem]


# ---------------------------------------------------------------------------
# Review response models
# ---------------------------------------------------------------------------
class ReviewNote(BaseModel):
    timing: str = Field(description="The timestamp of the cue being noted.")
    note: str = Field(description="The human-readable note about the cleanup choice.")


class Concern(BaseModel):
    timing: str = Field(description="The timestamp of the cue where the concern occurs.")
    issue_type: str = Field(description="The type of issue (e.g. 'untranslated', 'semantic', 'speaker').")
    severity: str = Field(description="The severity of the issue ('low', 'medium', 'high').")
    why_weird: str = Field(description="Why this specifically needs attention.")
    suggested_action: str = Field(description="What a human should do to fix it.")


class ReviewResponse(BaseModel):
    summary: str = Field(description="A brief summary of the overall transcript quality.")
    review_notes: list[ReviewNote] = Field(description="A list of noteworthy cleanup choices.")
    concerns: list[Concern] = Field(description="A list of suspicious places needing human attention.")
    needs_human_review: bool = Field(description="Whether a human should review this file.")


# ---------------------------------------------------------------------------
# Output path bundle
# ---------------------------------------------------------------------------
DEFAULT_AUDIO_EXTENSION = ".mp3"
DEFAULT_MANIFEST_NAME = "_anumodana_review_manifest.csv"
DEFAULT_COLLECTION_NAME = "Ajahn Wade Recordings"
DEFAULT_REVISION_DIR_NAME = "Transcript Revision"


def _revision_dir(root: Path) -> Path:
    return root.parent / DEFAULT_REVISION_DIR_NAME


def _revision_path(root: Path, source_path: Path, filename: str) -> Path:
    relative_parent = source_path.relative_to(root).parent
    return _revision_dir(root) / relative_parent / filename


class OutputPaths(BaseModel, frozen=True):
    """All derived output paths for a single source file."""

    audio: Path
    transcript: Path
    raw_vtt: Path
    cleaned_vtt: Path
    review_json: Path
    review_md: Path

    @classmethod
    def from_source(cls, root: Path, source_path: Path) -> OutputPaths:
        """Compute all output paths for a source file within a Trimmed root."""
        stem = source_path.stem
        audio = (
            source_path
            if source_path.suffix.lower() == DEFAULT_AUDIO_EXTENSION
            else source_path.with_suffix(DEFAULT_AUDIO_EXTENSION)
        )
        return cls(
            audio=audio,
            transcript=source_path.with_suffix(".txt"),
            raw_vtt=_revision_path(root, source_path, f"{stem}.parakeet.raw.vtt"),
            cleaned_vtt=_revision_path(root, source_path, f"{stem}.vtt"),
            review_json=_revision_path(root, source_path, f"{stem}.review.json"),
            review_md=_revision_path(root, source_path, f"{stem}.review.md"),
        )


# ---------------------------------------------------------------------------
# Standalone output path helpers (for the standalone fix / review commands)
# ---------------------------------------------------------------------------
def fixer_output_path(input_path: Path) -> Path:
    """Default output path for standalone ``anumodana fix``."""
    return input_path.with_name(f"{input_path.stem}.fixer.vtt")


def review_json_standalone_path(cleaned_vtt_path: Path) -> Path:
    return cleaned_vtt_path.with_name(f"{cleaned_vtt_path.stem}.review.json")


def review_md_standalone_path(cleaned_vtt_path: Path) -> Path:
    return cleaned_vtt_path.with_name(f"{cleaned_vtt_path.stem}.review.md")


def resolve_manifest_path(root: Path, manifest_arg: str) -> Path:
    if manifest_arg:
        return Path(manifest_arg).expanduser().resolve()
    return _revision_dir(root) / DEFAULT_MANIFEST_NAME
