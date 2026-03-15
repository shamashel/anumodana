from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_correct_vtt import (
    DEFAULT_GLOSSARY_FILES,
    DEFAULT_MODEL as DEFAULT_QWEN_MODEL,
    DEFAULT_OLLAMA_URL,
    correct_vtt_file,
    unload_ollama_model,
)
from review_vtt import (
    DEFAULT_CONTEXT_WINDOW as REVIEW_CONTEXT_WINDOW_DEFAULT,
    DEFAULT_TEMPERATURE as REVIEW_TEMPERATURE_DEFAULT,
    render_review_markdown,
    review_transcripts,
)

DEFAULT_ROOT = Path.home() / "Downloads" / "Trimmed"
DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_CHUNK_SECONDS = 120
DEFAULT_QWEN_BATCH_SIZE = 16
DEFAULT_QWEN_TEMPERATURE = 0.1
DEFAULT_QWEN_CONTEXT_WINDOW = 8192
DEFAULT_REVIEW_TEMPERATURE = REVIEW_TEMPERATURE_DEFAULT
DEFAULT_REVIEW_CONTEXT_WINDOW = REVIEW_CONTEXT_WINDOW_DEFAULT
DEFAULT_MANIFEST_NAME = "_anumodana_review_manifest.csv"
SUPPORTED_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".wav",
    ".webm",
}
VIDEO_EXTENSIONS = {".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
SOURCE_PRIORITY = {
    ".mp4": 0,
    ".mkv": 1,
    ".mov": 2,
    ".webm": 3,
    ".m4v": 4,
    ".mpeg": 5,
    ".mpg": 6,
    ".m4a": 7,
    ".mp3": 8,
    ".flac": 9,
    ".aac": 10,
    ".wav": 11,
}
EXCLUDED_DIR_NAMES = {".git", ".venv", ".uv-cache", ".uv-python", "__pycache__"}


@dataclass(frozen=True)
class Job:
    source_path: Path
    wav_path: Path
    raw_vtt_path: Path
    vtt_path: Path
    review_json_path: Path
    review_md_path: Path
    needs_wav: bool
    needs_raw_vtt: bool
    needs_vtt: bool
    needs_review: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk a tree, create same-name WAV files, transcribe with Parakeet v3, and clean VTTs with Qwen."
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
        help="Rebuild existing WAV and VTT outputs instead of skipping them.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help="Chunk long WAV files into this many seconds before Parakeet transcription.",
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
    return parser.parse_args()


def ensure_ffmpeg_on_path() -> str | None:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None

    ffmpeg_root = Path(local_app_data) / "Programs" / "ffmpeg"
    if not ffmpeg_root.exists():
        return None

    bin_directories = [path for path in ffmpeg_root.glob("*/bin") if path.is_dir()]
    if not bin_directories:
        return None

    newest_bin = max(bin_directories, key=lambda path: path.stat().st_mtime)
    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    if str(newest_bin) not in path_parts:
        os.environ["PATH"] = os.pathsep.join([str(newest_bin), *path_parts])

    return str(newest_bin)


def clean_caption_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def format_vtt_time(seconds: float) -> str:
    total_milliseconds = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def choose_source(candidates: list[Path]) -> Path:
    return min(
        candidates,
        key=lambda path: (
            SOURCE_PRIORITY.get(path.suffix.lower(), 99),
            path.suffix.lower(),
            path.name.lower(),
        ),
    )


def raw_vtt_output_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}.parakeet.raw.vtt")


def review_json_output_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}.review.json")


def review_md_output_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}.review.md")


def discover_jobs(root: Path, overwrite: bool, run_review: bool) -> tuple[list[Job], int]:
    grouped: dict[tuple[Path, str], list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if path.name.endswith(".whisper-input.wav"):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        grouped.setdefault((path.parent, path.stem), []).append(path)

    jobs: list[Job] = []
    skipped = 0

    for candidates in grouped.values():
        source_path = choose_source(candidates)
        wav_path = source_path if source_path.suffix.lower() == ".wav" else source_path.with_suffix(".wav")
        raw_vtt_path = raw_vtt_output_path(source_path)
        vtt_path = source_path.with_suffix(".vtt")
        review_json_path = review_json_output_path(source_path)
        review_md_path = review_md_output_path(source_path)
        has_wav = wav_path.exists()
        has_raw_vtt = raw_vtt_path.exists()
        has_vtt = vtt_path.exists()
        has_review = review_json_path.exists() and review_md_path.exists()

        if not overwrite and has_wav and has_raw_vtt and has_vtt and (has_review or not run_review):
            skipped += 1
            continue

        jobs.append(
            Job(
                source_path=source_path,
                wav_path=wav_path,
                raw_vtt_path=raw_vtt_path,
                vtt_path=vtt_path,
                review_json_path=review_json_path,
                review_md_path=review_md_path,
                needs_wav=overwrite or not has_wav,
                needs_raw_vtt=overwrite or not has_raw_vtt,
                needs_vtt=overwrite or not has_vtt,
                needs_review=run_review and (overwrite or not has_review),
            )
        )

    jobs.sort(key=lambda job: str(job.source_path).lower())
    return jobs, skipped


def extract_wav(source_path: Path, wav_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise FileNotFoundError(
            "FFmpeg was not found. Install an FFmpeg shared build or add ffmpeg.exe to PATH."
        )

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stdout.strip() or f"ffmpeg exited with code {process.returncode}")
    if not wav_path.exists():
        raise RuntimeError("ffmpeg finished without creating the expected WAV file.")


def get_media_duration_seconds(path: Path) -> float:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise FileNotFoundError(
            "FFprobe was not found. Install an FFmpeg shared build or add ffprobe.exe to PATH."
        )

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stdout.strip() or f"ffprobe exited with code {process.returncode}")

    try:
        return float(process.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe did not return a usable duration for {path}") from exc


def build_vtt_entries(hypothesis: object, offset_seconds: float = 0.0) -> list[tuple[float, float, str]]:
    timestamp_data = getattr(hypothesis, "timestamp", None)
    if not isinstance(timestamp_data, dict):
        return []

    raw_segments = timestamp_data.get("segment")
    entries: list[tuple[float, float, str]] = []
    if isinstance(raw_segments, list):
        for segment in raw_segments:
            if not isinstance(segment, dict):
                continue
            text = clean_caption_text(segment.get("segment"))
            if not text:
                continue
            start = float(segment.get("start", 0.0)) + offset_seconds
            end = float(segment.get("end", segment.get("start", 0.0) + 0.5)) + offset_seconds
            if end <= start:
                end = start + 0.5
            entries.append((start, end, text))

    if entries:
        return entries

    words = timestamp_data.get("word")
    if isinstance(words, list):
        collected_text: list[str] = []
        start = None
        end = None
        for word in words:
            if not isinstance(word, dict):
                continue
            text = clean_caption_text(word.get("word"))
            if not text:
                continue
            if start is None:
                start = float(word.get("start", 0.0)) + offset_seconds
            end = float(word.get("end", word.get("start", 0.0) + 0.5)) + offset_seconds
            collected_text.append(text)
        if collected_text and start is not None and end is not None:
            return [(start, max(end, start + 0.5), " ".join(collected_text))]

    text = clean_caption_text(getattr(hypothesis, "text", ""))
    if text:
        return [(offset_seconds, offset_seconds + 1.0, text)]
    return []


def write_vtt_entries(entries: list[tuple[float, float, str]], vtt_path: Path) -> None:
    if not entries:
        raise ValueError("Parakeet did not return any usable timestamped transcript segments.")

    lines: list[str] = ["WEBVTT", ""]
    for index, (start, end, text) in enumerate(entries, start=1):
        lines.append(str(index))
        lines.append(f"{format_vtt_time(start)} --> {format_vtt_time(end)}")
        lines.append(text)
        lines.append("")

    with vtt_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def iter_chunk_ranges(duration_seconds: float, chunk_seconds: int) -> list[tuple[float, float]]:
    if duration_seconds <= 0:
        return [(0.0, float(max(chunk_seconds, 1)))]

    chunk_length = float(max(chunk_seconds, 1))
    ranges: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_seconds:
        length = min(chunk_length, duration_seconds - start)
        ranges.append((start, length))
        start += chunk_length
    return ranges


def extract_chunk(source_wav: Path, chunk_path: Path, start_seconds: float, duration_seconds: float) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise FileNotFoundError(
            "FFmpeg was not found. Install an FFmpeg shared build or add ffmpeg.exe to PATH."
        )

    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(source_wav),
        "-acodec",
        "copy",
        str(chunk_path),
    ]
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stdout.strip() or f"ffmpeg exited with code {process.returncode}")
    if not chunk_path.exists():
        raise RuntimeError("ffmpeg finished without creating the expected chunk WAV file.")


def prepare_wav(job: Job) -> Path:
    if job.source_path.suffix.lower() == ".wav":
        return job.source_path
    if job.needs_wav:
        extract_wav(job.source_path, job.wav_path)
    return job.wav_path


def load_model(model_name: str) -> object:
    import nemo.collections.asr as nemo_asr
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name}", flush=True)
    print(f"Using device: {device}", flush=True)
    started = time.perf_counter()
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    model = model.to(device)
    print(f"Model ready in {time.perf_counter() - started:.2f}s", flush=True)
    return model


def transcribe_wav(model: object, wav_path: Path) -> object:
    original_cleanup = tempfile.TemporaryDirectory.cleanup

    def cleanup_ignoring_windows_manifest_lock(tempdir: object) -> None:
        try:
            original_cleanup(tempdir)
        except PermissionError:
            pass

    tempfile.TemporaryDirectory.cleanup = cleanup_ignoring_windows_manifest_lock
    try:
        outputs = model.transcribe([str(wav_path)], timestamps=True, batch_size=1)
    finally:
        tempfile.TemporaryDirectory.cleanup = original_cleanup

    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Parakeet returned no transcription output.")
    return outputs[0]


def transcribe_wav_to_entries(model: object, wav_path: Path, chunk_seconds: int) -> list[tuple[float, float, str]]:
    duration_seconds = get_media_duration_seconds(wav_path)
    if duration_seconds <= chunk_seconds:
        return build_vtt_entries(transcribe_wav(model, wav_path))

    entries: list[tuple[float, float, str]] = []
    with TemporaryDirectory(prefix="parakeet_chunks_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        for chunk_index, (start_seconds, length_seconds) in enumerate(
            iter_chunk_ranges(duration_seconds, chunk_seconds),
            start=1,
        ):
            chunk_path = temp_dir_path / f"chunk_{chunk_index:04d}.wav"
            print(
                f"  Chunk {chunk_index}: {start_seconds:.1f}s -> {start_seconds + length_seconds:.1f}s",
                flush=True,
            )
            extract_chunk(wav_path, chunk_path, start_seconds, length_seconds)
            hypothesis = transcribe_wav(model, chunk_path)
            entries.extend(build_vtt_entries(hypothesis, offset_seconds=start_seconds))

    return entries


def build_glossary_paths(args: argparse.Namespace) -> list[Path]:
    glossary_paths = [] if args.no_default_glossaries else list(DEFAULT_GLOSSARY_FILES)
    glossary_paths.extend(Path(path).expanduser().resolve() for path in args.glossary_file)
    return glossary_paths


def resolve_manifest_path(root: Path, manifest_arg: str) -> Path:
    if manifest_arg:
        return Path(manifest_arg).expanduser().resolve()
    return root / DEFAULT_MANIFEST_NAME


def load_review_metadata(review_json_path: Path) -> dict[str, object]:
    if not review_json_path.exists():
        return {}
    try:
        parsed = json.loads(review_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_manifest_rows(root: Path) -> list[dict[str, str]]:
    grouped: dict[tuple[Path, str], list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if path.name.endswith(".whisper-input.wav"):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        grouped.setdefault((path.parent, path.stem), []).append(path)

    rows: list[dict[str, str]] = []
    for candidates in grouped.values():
        source_path = choose_source(candidates)
        wav_path = source_path if source_path.suffix.lower() == ".wav" else source_path.with_suffix(".wav")
        raw_vtt_path = raw_vtt_output_path(source_path)
        cleaned_vtt_path = source_path.with_suffix(".vtt")
        review_json_path = review_json_output_path(source_path)
        review_md_path = review_md_output_path(source_path)
        review = load_review_metadata(review_json_path)
        review_notes = review.get("review_notes")
        concerns = review.get("concerns")
        rows.append(
            {
                "source_path": str(source_path),
                "wav_path": str(wav_path) if wav_path.exists() else "",
                "raw_vtt_path": str(raw_vtt_path) if raw_vtt_path.exists() else "",
                "cleaned_vtt_path": str(cleaned_vtt_path) if cleaned_vtt_path.exists() else "",
                "review_json_path": str(review_json_path) if review_json_path.exists() else "",
                "review_md_path": str(review_md_path) if review_md_path.exists() else "",
                "needs_human_review": (
                    "true" if isinstance(review.get("needs_human_review"), bool) and review.get("needs_human_review") else "false"
                    if review
                    else ""
                ),
                "review_note_count": str(len(review_notes) if isinstance(review_notes, list) else 0) if review else "",
                "concern_count": str(len(concerns) if isinstance(concerns, list) else 0) if review else "",
                "summary": str(review.get("summary", "")).replace("\n", " ").strip() if review else "",
            }
        )

    rows.sort(key=lambda row: row["source_path"].lower())
    return rows


def write_manifest_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "source_path",
        "wav_path",
        "raw_vtt_path",
        "cleaned_vtt_path",
        "review_json_path",
        "review_md_path",
        "needs_human_review",
        "review_note_count",
        "concern_count",
        "summary",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def release_parakeet_model(model: object | None) -> None:
    if model is None:
        return

    try:
        del model
        gc.collect()
    finally:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            return


def main() -> int:
    args = parse_args()
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
        write_manifest_csv(manifest_path, build_manifest_rows(root))
        print("Nothing to do.", flush=True)
        return 0

    if args.dry_run:
        for index, job in enumerate(jobs, start=1):
            print(f"[{index}] {job.source_path}", flush=True)
            print(f"    wav: {job.wav_path} ({'build' if job.needs_wav else 'reuse'})", flush=True)
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
    glossary_paths = build_glossary_paths(args)
    try:
        model = load_model(args.model_name)
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
                wav_path = prepare_wav(job)
                print(f"WAV: {wav_path}", flush=True)
                started = time.perf_counter()
                if job.needs_raw_vtt or job.needs_vtt:
                    entries = transcribe_wav_to_entries(model, wav_path, args.chunk_seconds)
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
                print(f"Transcription time: {elapsed:.2f}s", flush=True)
            except Exception as exc:
                failures += 1
                print(f"ERROR: {exc}", flush=True)
    finally:
        write_manifest_csv(manifest_path, build_manifest_rows(root))
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


if __name__ == "__main__":
    raise SystemExit(main())
