from __future__ import annotations

import gc
import tempfile
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from anumodana.ffmpeg import extract_chunk_wav, get_media_duration_seconds
from anumodana.transcript import VttEntry, build_vtt_entries


DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_CHUNK_SECONDS = 120


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

    # NeMo can leave a temporary manifest handle open on Windows.
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


def transcribe_audio_to_entries(
    model: object,
    source_media: Path,
    chunk_seconds: int,
) -> list[VttEntry]:
    duration_seconds = get_media_duration_seconds(source_media)
    entries: list[VttEntry] = []
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
            extract_chunk_wav(source_media, chunk_path, start_seconds, length_seconds)
            hypothesis = transcribe_wav(model, chunk_path)
            entries.extend(build_vtt_entries(hypothesis, offset_seconds=start_seconds))

    return entries


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
