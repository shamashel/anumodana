from __future__ import annotations

import contextlib
import gc
import io
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from anumodana.ffmpeg import extract_chunk_wav, get_media_duration_seconds
from anumodana.transcript import VttEntry, build_vtt_entries


DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_CHUNK_SECONDS = 120
QUIET_OUTPUT_PATTERNS = (
    re.compile(r"^\[NeMo [IW] "),
    re.compile(r"^W\d{4} "),
    re.compile(r"^OneLogger:"),
    re.compile(r"^No exporters were provided\."),
    re.compile(r"^Transcribing:"),
)


class _FilteredOutput(io.TextIOBase):
    def __init__(self, target: io.TextIOBase, patterns: tuple[re.Pattern[str], ...]) -> None:
        self._target = target
        self._patterns = patterns
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""
        self._target.flush()

    def _emit(self, line: str) -> None:
        if any(pattern.search(line) for pattern in self._patterns):
            return
        self._target.write(line + "\n")


def _configure_nemo_logging(verbose: bool) -> None:
    import nemo.utils

    nemo_level = nemo.utils.logging.INFO if verbose else nemo.utils.logging.ERROR
    nemo.utils.logging.set_verbosity(nemo_level)
    for logger_name in ("nemo_logger", "NeMo", "torch.distributed.elastic.multiprocessing.redirects"):
        logging.getLogger(logger_name).setLevel(logging.ERROR if not verbose else logging.INFO)


@contextlib.contextmanager
def _maybe_quiet_external_output(verbose: bool):
    if verbose:
        yield
        return

    filtered_stdout = _FilteredOutput(sys.stdout, QUIET_OUTPUT_PATTERNS)
    filtered_stderr = _FilteredOutput(sys.stderr, QUIET_OUTPUT_PATTERNS)
    with contextlib.redirect_stdout(filtered_stdout), contextlib.redirect_stderr(filtered_stderr):
        yield
    filtered_stdout.flush()
    filtered_stderr.flush()


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


def load_model(model_name: str, *, verbose: bool = False) -> object:
    with _maybe_quiet_external_output(verbose):
        import nemo.collections.asr as nemo_asr
        import torch

        _configure_nemo_logging(verbose)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model: {model_name}", flush=True)
        print(f"Using device: {device}", flush=True)
        started = time.perf_counter()
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        model = model.to(device)
        print(f"Model ready in {time.perf_counter() - started:.2f}s", flush=True)
        return model


def transcribe_wav(model: object, wav_path: Path, *, verbose: bool = False) -> object:
    original_cleanup = tempfile.TemporaryDirectory.cleanup

    # NeMo can leave a temporary manifest handle open on Windows.
    def cleanup_ignoring_windows_manifest_lock(tempdir: object) -> None:
        try:
            original_cleanup(tempdir)
        except PermissionError:
            pass

    tempfile.TemporaryDirectory.cleanup = cleanup_ignoring_windows_manifest_lock
    try:
        with _maybe_quiet_external_output(verbose):
            outputs = model.transcribe([str(wav_path)], timestamps=True, batch_size=1, verbose=verbose)
    finally:
        tempfile.TemporaryDirectory.cleanup = original_cleanup

    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Parakeet returned no transcription output.")
    return outputs[0]


def transcribe_audio_to_entries(
    model: object,
    source_media: Path,
    chunk_seconds: int,
    *,
    verbose: bool = False,
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
            hypothesis = transcribe_wav(model, chunk_path, verbose=verbose)
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
