"""CSV manifest helpers for pipeline and review output."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from anumodana.helpers.models import OutputPaths, ReviewResponse


PIPELINE_MANIFEST_FIELDNAMES = [
    "source_path",
    "audio_path",
    "transcript_path",
    "raw_vtt_path",
    "cleaned_vtt_path",
    "review_json_path",
    "review_md_path",
    "needs_human_review",
    "review_note_count",
    "concern_count",
    "summary",
]

REVIEW_MANIFEST_FIELDNAMES = [
    "raw_vtt_path",
    "cleaned_vtt_path",
    "review_json_path",
    "review_md_path",
    "needs_human_review",
    "review_note_count",
    "concern_count",
    "summary",
]


# ---------------------------------------------------------------------------
# Review metadata loading
# ---------------------------------------------------------------------------
def load_review_metadata(review_json_path: Path) -> ReviewResponse | None:
    """Load and validate review JSON, returning ``None`` on any failure."""
    if not review_json_path.exists():
        return None
    try:
        parsed = json.loads(review_json_path.read_text(encoding="utf-8"))
        return ReviewResponse.model_validate(parsed)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shared review field extraction (deduplicates row builders)
# ---------------------------------------------------------------------------
def _review_fields(review: ReviewResponse | None) -> dict[str, str]:
    if review is None:
        return {
            "needs_human_review": "",
            "review_note_count": "",
            "concern_count": "",
            "summary": "",
        }
    return {
        "needs_human_review": "true" if review.needs_human_review else "false",
        "review_note_count": str(len(review.review_notes)),
        "concern_count": str(len(review.concerns)),
        "summary": review.summary.replace("\n", " ").strip(),
    }


def _path_field(path: Path, *, require_exists: bool = False) -> str:
    if require_exists and not path.exists():
        return ""
    return path.parent.name


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------
def build_review_manifest_row(
    *,
    outputs: OutputPaths,
    review: ReviewResponse | None,
) -> dict[str, str]:
    return {
        "raw_vtt_path": outputs.raw_vtt.parent.name,
        "cleaned_vtt_path": outputs.cleaned_vtt.parent.name,
        "review_json_path": outputs.review_json.parent.name,
        "review_md_path": outputs.review_md.parent.name,
        **_review_fields(review),
    }


def build_pipeline_manifest_row(
    *,
    source_path: Path,
    outputs: OutputPaths,
    review: ReviewResponse | None,
) -> dict[str, str]:
    return {
        "source_path": source_path.parent.name,
        "audio_path": _path_field(outputs.audio, require_exists=True),
        "transcript_path": _path_field(outputs.transcript, require_exists=True),
        "raw_vtt_path": _path_field(outputs.raw_vtt, require_exists=True),
        "cleaned_vtt_path": _path_field(outputs.cleaned_vtt, require_exists=True),
        "review_json_path": _path_field(outputs.review_json, require_exists=True),
        "review_md_path": _path_field(outputs.review_md, require_exists=True),
        **_review_fields(review),
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def append_manifest_row(
    csv_path: Path,
    row: dict[str, str],
    *,
    fieldnames: Sequence[str],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_manifest_csv(
    csv_path: Path,
    rows: list[dict[str, str]],
    *,
    fieldnames: Sequence[str],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
