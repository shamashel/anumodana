"""Pipeline orchestration — ties together transcription, fixer, and review."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from ollama import Client

from anumodana.helpers.config import AnumodanaConfig, OllamaConfig
from anumodana.commands.fix import fix_vtt_file
from anumodana.helpers.ffmpeg import ensure_ffmpeg_on_path, extract_audio_copy
from anumodana.helpers.glossary import build_glossary_paths
from anumodana.helpers.jobs import Job, cleanup_transient_artifacts, discover_jobs, iter_preferred_sources
from anumodana.helpers.manifest import (
    PIPELINE_MANIFEST_FIELDNAMES,
    build_pipeline_manifest_row,
    load_review_metadata,
    write_manifest_csv,
)
from anumodana.helpers.models import DEFAULT_AUDIO_EXTENSION, DEFAULT_COLLECTION_NAME, OutputPaths, resolve_manifest_path
from anumodana.helpers.ollama import build_client, unload_model
from anumodana.helpers.parakeet import load_model, release_parakeet_model, transcribe_audio_to_entries
from anumodana.commands.review import render_review_markdown, review_transcripts
from anumodana.helpers.transcript import write_plain_text_from_vtt, write_vtt_entries

logger = logging.getLogger("anumodana")

DEFAULT_ROOT = Path.home() / "Downloads" / DEFAULT_COLLECTION_NAME
TRIMMED_DIR_NAME = "Trimmed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prepare_audio(job: Job) -> Path:
    if job.source_path.suffix.lower() == DEFAULT_AUDIO_EXTENSION:
        return job.source_path
    if job.needs_audio:
        extract_audio_copy(job.source_path, job.outputs.audio)
    return job.outputs.audio


def build_manifest_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source_path in iter_preferred_sources(root):
        outputs = OutputPaths.from_source(root, source_path)
        review = load_review_metadata(outputs.review_json)
        rows.append(build_pipeline_manifest_row(source_path=source_path, outputs=outputs, review=review))
    rows.sort(key=lambda row: row["source_path"].lower())
    return rows


def discover_trimmed_roots(root: Path) -> list[Path]:
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


# ---------------------------------------------------------------------------
# Pipeline entry points (called by cli.py)
# ---------------------------------------------------------------------------
def run_pipeline(
    *,
    config: AnumodanaConfig,
    root: Path,
    overwrite: bool = False,
    limit: int = 0,
    dry_run: bool = False,
    skip_fixer: bool = False,
    skip_review: bool = False,
    glossary_extra: list[str] | None = None,
    no_default_glossaries: bool = False,
    manifest_path_arg: str = "",
    keep_models_loaded: bool = False,
    verbose: bool = False,
) -> int:
    """Run the full pipeline (transcribe → fix → review)."""
    is_cloud = config.mode == "cloud"
    if is_cloud:
        logger.info("Using Ollama Cloud for fixer and review models.")
    else:
        logger.info("Using local Ollama.")

    root = root.expanduser().resolve()
    if not root.exists():
        logger.error("Root does not exist: %s", root)
        return 1

    run_review = not skip_review
    if skip_fixer and run_review:
        logger.info("Review pass disabled because fixer was skipped.")
        run_review = False

    ffmpeg_bin = ensure_ffmpeg_on_path()
    if ffmpeg_bin:
        logger.info("Using FFmpeg from: %s", ffmpeg_bin)

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
        logger.info("  Manifest: %s", manifest_paths[trimmed_root])

    if not all_jobs:
        for trimmed_root in trimmed_roots:
            write_manifest_csv(
                manifest_paths[trimmed_root],
                build_manifest_rows(trimmed_root),
                fieldnames=PIPELINE_MANIFEST_FIELDNAMES,
            )
        logger.info("Nothing to do.")
        return 0

    if dry_run:
        for index, (_, job) in enumerate(all_jobs, start=1):
            logger.info("[%d] %s", index, job.source_path)
            logger.info("    audio: %s (%s)", job.outputs.audio, "build" if job.needs_audio else "reuse")
            logger.info("    transcript: %s (%s)", job.outputs.transcript, "build" if job.needs_transcript else "reuse")
            logger.info("    raw_vtt: %s (%s)", job.outputs.raw_vtt, "build" if job.needs_raw_vtt else "reuse")
            vtt_mode = "build" if job.needs_cleaned_vtt else "reuse"
            if not skip_fixer:
                vtt_mode = f"{vtt_mode} -> fixer"
            logger.info("    cleaned_vtt: %s (%s)", job.outputs.cleaned_vtt, vtt_mode)
            review_mode = "skip"
            if run_review:
                review_mode = "build" if job.needs_review else "reuse"
            logger.info("    review_json: %s (%s)", job.outputs.review_json, review_mode)
            logger.info("    review_md: %s (%s)", job.outputs.review_md, review_mode)
        return 0

    # Build Ollama clients
    fixer_client: Client | None = None
    review_client: Client | None = None
    if not skip_fixer:
        fixer_client = build_client(config.fixer.ollama, api_key=config.api_key)
    if run_review:
        review_client = build_client(config.review.ollama, api_key=config.api_key)

    model = None
    failures = 0
    glossary_paths = build_glossary_paths(
        glossary_extra or [],
        include_defaults=not no_default_glossaries,
    )
    try:
        model = load_model(config.transcription.model, verbose=verbose)
        if not skip_fixer:
            logger.info("Fixer model: %s", config.fixer.ollama.model)
            if glossary_paths:
                logger.info("Fixer glossaries:")
                for path in glossary_paths:
                    logger.info("  %s", path)
        if run_review:
            logger.info("Review model: %s", config.review.ollama.model)

        for index, (_, job) in enumerate(all_jobs, start=1):
            logger.info("")
            logger.info("[%d/%d] %s", index, len(all_jobs), job.source_path)
            try:
                audio_path = prepare_audio(job)
                logger.info("Audio: %s", audio_path)
                started = time.perf_counter()
                if job.needs_raw_vtt or job.needs_cleaned_vtt or job.needs_transcript:
                    job.outputs.raw_vtt.parent.mkdir(parents=True, exist_ok=True)
                    job.outputs.cleaned_vtt.parent.mkdir(parents=True, exist_ok=True)
                    job.outputs.review_json.parent.mkdir(parents=True, exist_ok=True)
                    entries = transcribe_audio_to_entries(
                        model,
                        audio_path,
                        config.transcription.chunk_seconds,
                        verbose=verbose,
                    )
                    write_vtt_entries(entries, job.outputs.raw_vtt)
                    if skip_fixer:
                        shutil.copyfile(job.outputs.raw_vtt, job.outputs.cleaned_vtt)
                    else:
                        fix_vtt_file(
                            job.outputs.raw_vtt,
                            client=fixer_client,
                            config=config.fixer.ollama,
                            output_path=job.outputs.cleaned_vtt,
                            glossary_paths=glossary_paths,
                            batch_size=config.fixer.batch_size,
                            max_batch_characters=config.fixer.max_batch_characters,
                            max_prompt_tokens=config.fixer.max_prompt_tokens,
                        )
                    write_plain_text_from_vtt(job.outputs.cleaned_vtt, job.outputs.transcript)
                else:
                    logger.info("Reusing existing raw, cleaned, and shareable transcripts.")
                if run_review:
                    review = review_transcripts(
                        raw_vtt_path=job.outputs.raw_vtt,
                        cleaned_vtt_path=job.outputs.cleaned_vtt,
                        client=review_client,
                        config=config.review.ollama,
                        glossary_paths=glossary_paths,
                    )
                    job.outputs.review_json.write_text(
                        json.dumps(review.model_dump(), indent=2, ensure_ascii=False),
                        encoding="utf-8",
                        newline="\n",
                    )
                    job.outputs.review_md.write_text(
                        render_review_markdown(
                            review,
                            raw_vtt_path=job.outputs.raw_vtt,
                            cleaned_vtt_path=job.outputs.cleaned_vtt,
                        ),
                        encoding="utf-8",
                        newline="\n",
                    )
                elapsed = time.perf_counter() - started
                logger.info("Wrote raw VTT: %s", job.outputs.raw_vtt)
                logger.info("Wrote cleaned VTT: %s", job.outputs.cleaned_vtt)
                logger.info("Wrote transcript: %s", job.outputs.transcript)
                if run_review:
                    logger.info("Wrote review JSON: %s", job.outputs.review_json)
                    logger.info("Wrote review markdown: %s", job.outputs.review_md)
                removed_artifacts = cleanup_transient_artifacts(job)
                for removed_path in removed_artifacts:
                    logger.info("Removed transient artifact: %s", removed_path)
                logger.info("Transcription time: %.2fs", elapsed)
            except Exception as exc:
                failures += 1
                logger.error("ERROR: %s", exc)
    finally:
        for trimmed_root in trimmed_roots:
            write_manifest_csv(
                manifest_paths[trimmed_root],
                build_manifest_rows(trimmed_root),
                fieldnames=PIPELINE_MANIFEST_FIELDNAMES,
            )
        if not keep_models_loaded:
            release_parakeet_model(model)
            if fixer_client and not skip_fixer:
                unload_model(fixer_client, config.fixer.ollama)
            if review_client and run_review:
                unload_model(review_client, config.review.ollama)

    logger.info("")
    logger.info("Completed: %d", len(all_jobs) - failures)
    logger.info("Failed: %d", failures)
    for trimmed_root in trimmed_roots:
        logger.info("Updated manifest: %s", manifest_paths[trimmed_root])
    return 1 if failures else 0
