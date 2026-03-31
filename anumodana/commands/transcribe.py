"""ASR-only transcription pass — no fixer, no review."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from anumodana.helpers.batch import JobProcessor, prepare_audio, run_batch_jobs
from anumodana.helpers.config import AnumodanaConfig
from anumodana.helpers.jobs import Job, cleanup_transient_artifacts
from anumodana.helpers.parakeet import (
    load_model,
    release_parakeet_model,
    transcribe_audio_to_entries,
)
from anumodana.helpers.transcript import write_plain_text_from_vtt, write_vtt_entries

logger = logging.getLogger("anumodana")


def run_transcribe(
    *,
    config: AnumodanaConfig,
    root: Path,
    overwrite: bool = False,
    limit: int = 0,
    dry_run: bool = False,
    keep_models_loaded: bool = False,
    verbose: bool = False,
) -> int:
    """Run ASR-only transcription on a batch of files."""
    model = None

    def process_transcribe(index: int, total: int, root: Path, job: Job) -> bool:
        if dry_run:
            logger.info("[%d] %s", index, job.source_path)
            logger.info("    audio: %s (%s)", job.outputs.audio, "build" if job.needs_audio else "reuse")
            logger.info("    transcript: %s (%s)", job.outputs.transcript, "build" if job.needs_transcript else "reuse")
            logger.info("    raw_vtt: %s (%s)", job.outputs.raw_vtt, "build" if job.needs_raw_vtt else "reuse")
            logger.info("    cleaned_vtt: (skip)")
            logger.info("    review_json: (skip)")
            logger.info("    review_md: (skip)")
            return True

        nonlocal model
        if model is None:
            model = load_model(config.transcription.model, verbose=verbose)

        logger.info("")
        logger.info("[%d/%d] %s", index, total, job.source_path)
        try:
            audio_path = prepare_audio(job)
            logger.info("Audio: %s", audio_path)
            started = time.perf_counter()

            if job.needs_raw_vtt or job.needs_transcript:
                job.outputs.raw_vtt.parent.mkdir(parents=True, exist_ok=True)
                entries = transcribe_audio_to_entries(
                    model,
                    audio_path,
                    config.transcription.chunk_seconds,
                    verbose=verbose,
                )
                write_vtt_entries(entries, job.outputs.raw_vtt)
                write_plain_text_from_vtt(job.outputs.raw_vtt, job.outputs.transcript)
            else:
                logger.info("Reusing existing raw transcript and text output.")

            elapsed = time.perf_counter() - started
            logger.info("Wrote raw VTT: %s", job.outputs.raw_vtt)
            logger.info("Wrote transcript: %s", job.outputs.transcript)

            removed_artifacts = cleanup_transient_artifacts(job)
            for removed_path in removed_artifacts:
                logger.info("Removed transient artifact: %s", removed_path)
            logger.info("Transcription time: %.2fs", elapsed)
            return True
        except Exception as exc:
            logger.error("ERROR: %s", exc)
            return False

    try:
        return run_batch_jobs(
            root=root,
            overwrite=overwrite,
            limit=limit,
            dry_run=dry_run,
            run_review=False,
            manifest_path_arg="",
            processor=process_transcribe,
            finalize_manifest=False,
        )
    finally:
        if not keep_models_loaded and model is not None:
            release_parakeet_model(model)
