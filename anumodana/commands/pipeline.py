"""Pipeline orchestration — ties together transcription, fixer, and review."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from ollama import Client

from anumodana.commands.fix import fix_vtt_file
from anumodana.helpers.batch import (
    JobProcessor,
    prepare_audio,
    run_batch_jobs,
)
from anumodana.helpers.config import AnumodanaConfig
from anumodana.helpers.ffmpeg import ensure_ffmpeg_on_path
from anumodana.helpers.glossary import build_glossary_paths
from anumodana.helpers.jobs import Job, cleanup_transient_artifacts
from anumodana.helpers.manifest import (
    PIPELINE_MANIFEST_FIELDNAMES,
    write_manifest_csv,
)
from anumodana.helpers.models import (
    DEFAULT_ROOT,
)
from anumodana.helpers.ollama import build_client, unload_model
from anumodana.helpers.parakeet import (
    load_model,
    release_parakeet_model,
    transcribe_audio_to_entries,
)
from anumodana.commands.review import render_review_markdown, review_transcripts
from anumodana.helpers.transcript import write_plain_text_from_vtt, write_vtt_entries

logger = logging.getLogger("anumodana")


def run_pipeline(
    *,
    config: AnumodanaConfig,
    root: Path,
    overwrite: bool = False,
    limit: int = 0,
    dry_run: bool = False,
    review: bool = False,
    glossary_extra: list[str] | None = None,
    no_default_glossaries: bool = False,
    manifest_path_arg: str = "",
    keep_models_loaded: bool = False,
    verbose: bool = False,
) -> int:
    """Run the transcription pipeline: transcribe → fix (optional review)."""
    is_cloud = config.mode == "cloud"
    if is_cloud:
        logger.info("Using Ollama Cloud for fixer and review models.")
    else:
        logger.info("Using local Ollama.")

    ffmpeg_bin = ensure_ffmpeg_on_path()
    if ffmpeg_bin:
        logger.info("Using FFmpeg from: %s", ffmpeg_bin)

    fixer_client = build_client(config.fixer.ollama, api_key=config.api_key)
    review_client = None
    if review:
        review_client = build_client(config.review.ollama, api_key=config.api_key)

    model = None
    glossary_paths = build_glossary_paths(
        glossary_extra or [],
        include_defaults=not no_default_glossaries,
    )

    def process_pipeline(index: int, total: int, root: Path, job: Job) -> bool:
        if dry_run:
            logger.info("[%d] %s", index, job.source_path)
            logger.info("    audio: %s (%s)", job.outputs.audio, "build" if job.needs_audio else "reuse")
            logger.info("    transcript: %s (%s)", job.outputs.transcript, "build" if job.needs_transcript else "reuse")
            logger.info("    raw_vtt: %s (%s)", job.outputs.raw_vtt, "build" if job.needs_raw_vtt else "reuse")
            vtt_mode = "build" if job.needs_cleaned_vtt else "reuse"
            vtt_mode = f"{vtt_mode} -> fixer"
            logger.info("    cleaned_vtt: %s (%s)", job.outputs.cleaned_vtt, vtt_mode)
            review_mode = "skip"
            if review:
                review_mode = "build" if job.needs_review else "reuse"
            logger.info("    review_json: %s (%s)", job.outputs.review_json, review_mode)
            logger.info("    review_md: %s (%s)", job.outputs.review_md, review_mode)
            return True

        nonlocal model
        if model is None:
            model = load_model(config.transcription.model, verbose=verbose)
            logger.info("Fixer model: %s", config.fixer.ollama.model)
            if glossary_paths:
                logger.info("Fixer glossaries:")
                for path in glossary_paths:
                    logger.info("  %s", path)
            if review:
                logger.info("Review model: %s", config.review.ollama.model)

        logger.info("")
        logger.info("[%d/%d] %s", index, total, job.source_path)
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

            if review:
                review_result = review_transcripts(
                    raw_vtt_path=job.outputs.raw_vtt,
                    cleaned_vtt_path=job.outputs.cleaned_vtt,
                    client=review_client,
                    config=config.review.ollama,
                    glossary_paths=glossary_paths,
                )
                job.outputs.review_json.write_text(
                    json.dumps(review_result.model_dump(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                    newline="\n",
                )
                job.outputs.review_md.write_text(
                    render_review_markdown(
                        review_result,
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
            if review:
                logger.info("Wrote review JSON: %s", job.outputs.review_json)
                logger.info("Wrote review markdown: %s", job.outputs.review_md)

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
            run_review=review,
            manifest_path_arg=manifest_path_arg,
            processor=process_pipeline,
            finalize_manifest=review,
        )
    finally:
        if not keep_models_loaded:
            if model is not None:
                release_parakeet_model(model)
            if fixer_client:
                unload_model(fixer_client, config.fixer.ollama)
            if review_client and review:
                unload_model(review_client, config.review.ollama)
