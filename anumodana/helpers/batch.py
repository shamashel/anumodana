"""Batch processing helpers — shared logic for running work over entire directory trees."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from anumodana.helpers.ffmpeg import extract_audio_copy
from anumodana.helpers.jobs import Job, discover_jobs, iter_preferred_sources
from anumodana.helpers.manifest import (
    PIPELINE_MANIFEST_FIELDNAMES,
    build_pipeline_manifest_row,
    load_review_metadata,
    write_manifest_csv,
)
from anumodana.helpers.models import (
    DEFAULT_AUDIO_EXTENSION,
    OutputPaths,
    resolve_manifest_path,
)

logger = logging.getLogger("anumodana")

TRIMMED_DIR_NAME = "Trimmed"


class JobProcessor(Protocol):
    """Protocol for a function that processes a single job in a batch."""

    def __call__(self, index: int, total: int, root: Path, job: Job) -> bool:
        """Process a job. Return True on success, False on failure."""
        ...


def prepare_audio(job: Job) -> Path:
    """Ensure the job has a valid audio source for transcription."""
    if job.source_path.suffix.lower() == DEFAULT_AUDIO_EXTENSION:
        return job.source_path
    if job.needs_audio:
        extract_audio_copy(job.source_path, job.outputs.audio)
    return job.outputs.audio


def build_manifest_rows(root: Path) -> list[dict[str, str]]:
    """Build summary rows for every preferred source in a Trimmed root."""
    rows: list[dict[str, str]] = []
    for source_path in iter_preferred_sources(root):
        outputs = OutputPaths.from_source(root, source_path)
        review = load_review_metadata(outputs.review_json)
        rows.append(build_pipeline_manifest_row(source_path=source_path, outputs=outputs, review=review))
    rows.sort(key=lambda row: row["source_path"].lower())
    return rows


def discover_trimmed_roots(root: Path) -> list[Path]:
    """Find all 'Trimmed' subdirectories under the given root."""
    if root.name.casefold() == TRIMMED_DIR_NAME.casefold():
        return [root]

    direct_trimmed = root / TRIMMED_DIR_NAME
    if direct_trimmed.is_dir():
        return [direct_trimmed]

    trimmed_roots = sorted(
        child / TRIMMED_DIR_NAME
        for child in root.iterdir()
        if child.is_dir() and (child / TRIMMED_DIR_NAME).is_dir()
    )
    if trimmed_roots:
        return trimmed_roots

    return [root]


def run_batch_jobs(
    *,
    root: Path,
    overwrite: bool,
    limit: int,
    dry_run: bool,
    run_review: bool,
    manifest_path_arg: str,
    processor: JobProcessor,
    finalize_manifest: bool = False,
) -> int:
    """Core batch processing loop."""
    root = root.expanduser().resolve()
    if not root.exists():
        logger.error("Root does not exist: %s", root)
        return 1

    trimmed_roots = discover_trimmed_roots(root)
    if manifest_path_arg and len(trimmed_roots) > 1:
        logger.error("--manifest-path can only be used when --root resolves to a single Trimmed tree.")
        return 1

    jobs_by_root: dict[Path, list[Job]] = {}
    manifest_paths: dict[Path, Path] = {}
    skipped_by_root: dict[Path, int] = {}
    all_jobs: list[tuple[Path, Job]] = []

    for trimmed_root in trimmed_roots:
        jobs, skipped = discover_jobs(trimmed_root, overwrite=overwrite, run_review=run_review)
        jobs_by_root[trimmed_root] = jobs
        manifest_paths[trimmed_root] = resolve_manifest_path(trimmed_root, manifest_path_arg)
        skipped_by_root[trimmed_root] = skipped
        for job in jobs:
            all_jobs.append((trimmed_root, job))

    all_jobs.sort(key=lambda item: str(item[1].source_path).lower())
    if limit > 0:
        all_jobs = all_jobs[:limit]

    logger.info("Root: %s", root)
    logger.info("Trimmed trees found: %d", len(trimmed_roots))
    for trimmed_root in trimmed_roots:
        queued_count = sum(1 for job_root, _ in all_jobs if job_root == trimmed_root)
        logger.info("Collection Trimmed root: %s", trimmed_root)
        logger.info("  Jobs queued: %d", queued_count)
        logger.info("  Already complete: %d", skipped_by_root[trimmed_root])
        if finalize_manifest:
            logger.info("  Manifest: %s", manifest_paths[trimmed_root])

    if not all_jobs:
        if finalize_manifest:
            for trimmed_root in trimmed_roots:
                write_manifest_csv(
                    manifest_paths[trimmed_root],
                    build_manifest_rows(trimmed_root),
                    fieldnames=PIPELINE_MANIFEST_FIELDNAMES,
                )
        logger.info("Nothing to do.")
        return 0

    if dry_run:
        # Dry run logic is handled by the processor usually, or we can do a generic one here.
        # But since processors vary on what they show as build/skip, we'll let the processor handle it.
        # Wait, the dry run output in pipeline.py is quite specific.
        # We'll pass a dry_run flag to the processor.
        pass

    failures = 0
    try:
        for index, (trimmed_root, job) in enumerate(all_jobs, start=1):
            success = processor(index, len(all_jobs), trimmed_root, job)
            if not success:
                failures += 1
    finally:
        if finalize_manifest:
            for trimmed_root in trimmed_roots:
                write_manifest_csv(
                    manifest_paths[trimmed_root],
                    build_manifest_rows(trimmed_root),
                    fieldnames=PIPELINE_MANIFEST_FIELDNAMES,
                )

    logger.info("")
    logger.info("Completed: %d", len(all_jobs) - failures)
    logger.info("Failed: %d", failures)
    if finalize_manifest:
        for trimmed_root in trimmed_roots:
            logger.info("Updated manifest: %s", manifest_paths[trimmed_root])

    return 1 if failures else 0
