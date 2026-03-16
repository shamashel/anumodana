from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path


PIPELINE_MANIFEST_FIELDNAMES = [
    "source_path",
    "audio_path",
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


def load_review_metadata(review_json_path: Path) -> dict[str, object]:
    if not review_json_path.exists():
        return {}
    try:
        parsed = json.loads(review_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_review_manifest_row(
    *,
    raw_vtt_path: Path,
    cleaned_vtt_path: Path,
    review_json_path: Path,
    review_md_path: Path,
    review: dict[str, object],
) -> dict[str, str]:
    review_notes = review.get("review_notes", [])
    concerns = review.get("concerns", [])
    return {
        "raw_vtt_path": str(raw_vtt_path),
        "cleaned_vtt_path": str(cleaned_vtt_path),
        "review_json_path": str(review_json_path),
        "review_md_path": str(review_md_path),
        "needs_human_review": "true" if review.get("needs_human_review") else "false",
        "review_note_count": str(len(review_notes) if isinstance(review_notes, list) else 0),
        "concern_count": str(len(concerns) if isinstance(concerns, list) else 0),
        "summary": str(review.get("summary", "")).replace("\n", " ").strip(),
    }


def build_pipeline_manifest_row(
    *,
    source_path: Path,
    audio_path: Path,
    raw_vtt_path: Path,
    cleaned_vtt_path: Path,
    review_json_path: Path,
    review_md_path: Path,
    review: dict[str, object],
) -> dict[str, str]:
    review_notes = review.get("review_notes")
    concerns = review.get("concerns")
    return {
        "source_path": str(source_path),
        "audio_path": str(audio_path) if audio_path.exists() else "",
        "raw_vtt_path": str(raw_vtt_path) if raw_vtt_path.exists() else "",
        "cleaned_vtt_path": str(cleaned_vtt_path) if cleaned_vtt_path.exists() else "",
        "review_json_path": str(review_json_path) if review_json_path.exists() else "",
        "review_md_path": str(review_md_path) if review_md_path.exists() else "",
        "needs_human_review": (
            "true" if isinstance(review.get("needs_human_review"), bool) and review.get("needs_human_review") else "false"
        )
        if review
        else "",
        "review_note_count": str(len(review_notes) if isinstance(review_notes, list) else 0) if review else "",
        "concern_count": str(len(concerns) if isinstance(concerns, list) else 0) if review else "",
        "summary": str(review.get("summary", "")).replace("\n", " ").strip() if review else "",
    }


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
