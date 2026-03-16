from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from anumodana.correction import correct_vtt_file
from anumodana.ffmpeg import ensure_ffmpeg_on_path, extract_audio_copy
from anumodana.glossary import build_glossary_paths
from anumodana.jobs import Job, cleanup_transient_artifacts, discover_jobs, iter_preferred_sources
from anumodana.manifest import (
    PIPELINE_MANIFEST_FIELDNAMES,
    build_pipeline_manifest_row,
    load_review_metadata,
    write_manifest_csv,
)
from anumodana.ollama import DEFAULT_MODEL as DEFAULT_QWEN_MODEL
from anumodana.ollama import DEFAULT_OLLAMA_URL, unload_ollama_model
from anumodana.output_paths import (
    DEFAULT_AUDIO_EXTENSION,
    audio_output_path,
    cleaned_vtt_output_path,
    raw_vtt_output_path,
    resolve_manifest_path,
    review_json_output_path,
    review_md_output_path,
)
from anumodana.parakeet import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_MODEL,
    load_model,
    release_parakeet_model,
    transcribe_audio_to_entries,
)
from anumodana.review import (
    DEFAULT_CONTEXT_WINDOW as REVIEW_CONTEXT_WINDOW_DEFAULT,
    DEFAULT_TEMPERATURE as REVIEW_TEMPERATURE_DEFAULT,
    render_review_markdown,
    review_transcripts,
)
from anumodana.transcript import write_vtt_entries


DEFAULT_ROOT = Path.home() / "Downloads" / "Trimmed"
DEFAULT_QWEN_BATCH_SIZE = 16
DEFAULT_QWEN_TEMPERATURE = 0.1
DEFAULT_QWEN_CONTEXT_WINDOW = 8192
DEFAULT_REVIEW_TEMPERATURE = REVIEW_TEMPERATURE_DEFAULT
DEFAULT_REVIEW_CONTEXT_WINDOW = REVIEW_CONTEXT_WINDOW_DEFAULT


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Walk a tree, create same-name MP3 audio copies, transcribe with Parakeet v3, and clean VTTs with Qwen."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Root directory to scan. Defaults to the Trimmed downloads tree.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="NeMo / Hugging Face model id to load once for the whole batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N jobs after discovery. 0 means no limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing audio and VTT outputs instead of skipping them.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help="Chunk audio into this many seconds before Parakeet transcription.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would run without extracting or transcribing.",
    )
    parser.add_argument(
        "--skip-qwen",
        action="store_true",
        help="Skip the local Qwen correction pass and keep raw Parakeet VTT output.",
    )
    parser.add_argument(
        "--qwen-model",
        default=DEFAULT_QWEN_MODEL,
        help="Ollama model name for the cleanup pass.",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama generate endpoint for the cleanup pass.",
    )
    parser.add_argument(
        "--qwen-batch-size",
        type=int,
        default=DEFAULT_QWEN_BATCH_SIZE,
        help="Number of subtitle cues per Ollama cleanup request. 0 means send the whole transcript in one request.",
    )
    parser.add_argument(
        "--qwen-temperature",
        type=float,
        default=DEFAULT_QWEN_TEMPERATURE,
        help="Sampling temperature for the Ollama cleanup pass.",
    )
    parser.add_argument(
        "--qwen-context-window",
        type=int,
        default=DEFAULT_QWEN_CONTEXT_WINDOW,
        help="num_ctx to send to Ollama for the cleanup pass.",
    )
    parser.add_argument(
        "--glossary-file",
        action="append",
        default=[],
        help="Additional glossary file to append for the Qwen cleanup pass.",
    )
    parser.add_argument(
        "--no-default-glossaries",
        action="store_true",
        help="Do not load the built-in glossary stack for the Qwen cleanup pass.",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip the structured review pass and do not write review JSON/markdown files.",
    )
    parser.add_argument(
        "--review-model",
        default=DEFAULT_QWEN_MODEL,
        help="Ollama model name for the structured review pass.",
    )
    parser.add_argument(
        "--review-temperature",
        type=float,
        default=DEFAULT_REVIEW_TEMPERATURE,
        help="Sampling temperature for the structured review pass.",
    )
    parser.add_argument(
        "--review-context-window",
        type=int,
        default=DEFAULT_REVIEW_CONTEXT_WINDOW,
        help="num_ctx to send to Ollama for the structured review pass.",
    )
    parser.add_argument(
        "--manifest-path",
        default="",
        help="Where to write the root-level review manifest CSV. Defaults to <root>\\_anumodana_review_manifest.csv.",
    )
    parser.add_argument(
        "--keep-models-loaded",
        action="store_true",
        help="Do not unload the Parakeet or Qwen models after the batch run finishes.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose library logs and transcribe progress from NeMo and related dependencies.",
    )
    return parser.parse_args(argv)


def prepare_audio(job: Job) -> Path:
    if job.source_path.suffix.lower() == DEFAULT_AUDIO_EXTENSION:
        return job.source_path
    if job.needs_audio:
        extract_audio_copy(job.source_path, job.audio_path)
    return job.audio_path


def build_manifest_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source_path in iter_preferred_sources(root):
        audio_path = audio_output_path(source_path)
        raw_vtt_path = raw_vtt_output_path(source_path)
        cleaned_vtt_path = cleaned_vtt_output_path(source_path)
        review_json_path = review_json_output_path(source_path)
        review_md_path = review_md_output_path(source_path)
        review = load_review_metadata(review_json_path)
        rows.append(
            build_pipeline_manifest_row(
                source_path=source_path,
                audio_path=audio_path,
                raw_vtt_path=raw_vtt_path,
                cleaned_vtt_path=cleaned_vtt_path,
                review_json_path=review_json_path,
                review_md_path=review_md_path,
                review=review,
            )
        )

    rows.sort(key=lambda row: row["source_path"].lower())
    return rows


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
) -> int:
    args = parse_args(argv, prog=prog)
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        return 1

    run_review = not args.skip_review
    if args.skip_qwen and run_review:
        print("Review pass disabled because --skip-qwen was requested.", flush=True)
        run_review = False

    ffmpeg_bin = ensure_ffmpeg_on_path()
    if ffmpeg_bin:
        print(f"Using FFmpeg from: {ffmpeg_bin}", flush=True)

    jobs, skipped = discover_jobs(root, overwrite=args.overwrite, run_review=run_review)
    if args.limit > 0:
        jobs = jobs[: args.limit]
    manifest_path = resolve_manifest_path(root, args.manifest_path)

    print(f"Root: {root}", flush=True)
    print(f"Jobs queued: {len(jobs)}", flush=True)
    print(f"Already complete: {skipped}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

    if not jobs:
        write_manifest_csv(
            manifest_path,
            build_manifest_rows(root),
            fieldnames=PIPELINE_MANIFEST_FIELDNAMES,
        )
        print("Nothing to do.", flush=True)
        return 0

    if args.dry_run:
        for index, job in enumerate(jobs, start=1):
            print(f"[{index}] {job.source_path}", flush=True)
            print(f"    audio: {job.audio_path} ({'build' if job.needs_audio else 'reuse'})", flush=True)
            print(f"    raw_vtt: {job.raw_vtt_path} ({'build' if job.needs_raw_vtt else 'reuse'})", flush=True)
            vtt_mode = "build" if job.needs_vtt else "reuse"
            if not args.skip_qwen:
                vtt_mode = f"{vtt_mode} -> qwen"
            print(f"    vtt: {job.vtt_path} ({vtt_mode})", flush=True)
            review_mode = "skip"
            if run_review:
                review_mode = "build" if job.needs_review else "reuse"
            print(f"    review_json: {job.review_json_path} ({review_mode})", flush=True)
            print(f"    review_md: {job.review_md_path} ({review_mode})", flush=True)
        return 0

    model = None
    failures = 0
    glossary_paths = build_glossary_paths(
        args.glossary_file,
        include_defaults=not args.no_default_glossaries,
    )
    try:
        model = load_model(args.model_name, verbose=args.verbose)
        if not args.skip_qwen:
            print(f"Qwen cleanup model: {args.qwen_model}", flush=True)
            if glossary_paths:
                print("Qwen glossaries:", flush=True)
                for path in glossary_paths:
                    print(f"  {path}", flush=True)
        if run_review:
            print(f"Review model: {args.review_model}", flush=True)

        for index, job in enumerate(jobs, start=1):
            print("", flush=True)
            print(f"[{index}/{len(jobs)}] {job.source_path}", flush=True)
            try:
                audio_path = prepare_audio(job)
                print(f"Audio: {audio_path}", flush=True)
                started = time.perf_counter()
                if job.needs_raw_vtt or job.needs_vtt:
                    entries = transcribe_audio_to_entries(
                        model,
                        audio_path,
                        args.chunk_seconds,
                        verbose=args.verbose,
                    )
                    write_vtt_entries(entries, job.raw_vtt_path)
                    if args.skip_qwen:
                        shutil.copyfile(job.raw_vtt_path, job.vtt_path)
                    else:
                        correct_vtt_file(
                            job.raw_vtt_path,
                            output_path=job.vtt_path,
                            glossary_paths=glossary_paths,
                            model=args.qwen_model,
                            ollama_url=args.ollama_url,
                            batch_size=args.qwen_batch_size,
                            temperature=args.qwen_temperature,
                            context_window=args.qwen_context_window,
                            progress=True,
                        )
                else:
                    print("Reusing existing raw and cleaned transcripts.", flush=True)
                if run_review:
                    review = review_transcripts(
                        raw_vtt_path=job.raw_vtt_path,
                        cleaned_vtt_path=job.vtt_path,
                        glossary_paths=glossary_paths,
                        model=args.review_model,
                        ollama_url=args.ollama_url,
                        temperature=args.review_temperature,
                        context_window=args.review_context_window,
                    )
                    job.review_json_path.write_text(
                        json.dumps(review, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                        newline="\n",
                    )
                    job.review_md_path.write_text(
                        render_review_markdown(
                            review,
                            raw_vtt_path=job.raw_vtt_path,
                            cleaned_vtt_path=job.vtt_path,
                        ),
                        encoding="utf-8",
                        newline="\n",
                    )
                elapsed = time.perf_counter() - started
                print(f"Wrote raw VTT: {job.raw_vtt_path}", flush=True)
                print(f"Wrote: {job.vtt_path}", flush=True)
                if run_review:
                    print(f"Wrote review JSON: {job.review_json_path}", flush=True)
                    print(f"Wrote review markdown: {job.review_md_path}", flush=True)
                removed_artifacts = cleanup_transient_artifacts(job)
                for removed_path in removed_artifacts:
                    print(f"Removed transient artifact: {removed_path}", flush=True)
                print(f"Transcription time: {elapsed:.2f}s", flush=True)
            except Exception as exc:
                failures += 1
                print(f"ERROR: {exc}", flush=True)
    finally:
        write_manifest_csv(
            manifest_path,
            build_manifest_rows(root),
            fieldnames=PIPELINE_MANIFEST_FIELDNAMES,
        )
        if not args.keep_models_loaded:
            release_parakeet_model(model)
            models_to_unload: list[str] = []
            if not args.skip_qwen:
                models_to_unload.append(args.qwen_model)
            if run_review:
                models_to_unload.append(args.review_model)
            for model_name in sorted(set(models_to_unload)):
                unload_ollama_model(model_name, args.ollama_url)

    print("", flush=True)
    print(f"Completed: {len(jobs) - failures}", flush=True)
    print(f"Failed: {failures}", flush=True)
    print(f"Updated manifest: {manifest_path}", flush=True)
    return 1 if failures else 0
