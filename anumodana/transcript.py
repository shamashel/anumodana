from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


VttEntry = tuple[float, float, str]


@dataclass(frozen=True)
class Cue:
    index: int
    timing: str
    text: str


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


def parse_vtt(path: Path) -> list[Cue]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cues: list[Cue] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "WEBVTT":
            i += 1
            continue
        if not line.isdigit():
            i += 1
            continue
        index = int(line)
        if i + 2 >= len(lines):
            break
        timing = lines[i + 1].rstrip()
        text_lines: list[str] = []
        i += 2
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1
        text = " ".join(text_lines).strip()
        cues.append(Cue(index=index, timing=timing, text=text))
    return cues


def render_vtt(
    cues: list[Cue],
    corrected_text: Mapping[int, str] | None = None,
) -> str:
    lines: list[str] = ["WEBVTT", ""]
    resolved_text = corrected_text or {}
    for cue in cues:
        lines.append(str(cue.index))
        lines.append(cue.timing)
        lines.append(resolved_text.get(cue.index, cue.text))
        lines.append("")
    return "\n".join(lines)


def render_plain_text(
    cues: list[Cue],
    corrected_text: Mapping[int, str] | None = None,
) -> str:
    resolved_text = corrected_text or {}
    lines = [clean_caption_text(resolved_text.get(cue.index, cue.text)) for cue in cues]
    return "\n".join(line for line in lines if line).strip() + "\n"


def build_vtt_entries(hypothesis: object, offset_seconds: float = 0.0) -> list[VttEntry]:
    timestamp_data = getattr(hypothesis, "timestamp", None)
    if not isinstance(timestamp_data, dict):
        return []

    raw_segments = timestamp_data.get("segment")
    entries: list[VttEntry] = []
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


def write_vtt_entries(entries: list[VttEntry], vtt_path: Path) -> None:
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


def write_plain_text_from_vtt(vtt_path: Path, transcript_path: Path) -> None:
    cues = parse_vtt(vtt_path)
    if not cues:
        raise ValueError(f"No VTT cues found in {vtt_path}")
    transcript_path.write_text(render_plain_text(cues), encoding="utf-8", newline="\n")
