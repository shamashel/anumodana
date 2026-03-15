from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_BATCH_CHARACTERS = 3200
DEFAULT_MAX_PROMPT_TOKENS = 2600
DEFAULT_GLOSSARY_FILES = [
    Path(__file__).with_name("glossaries") / "core_chants.txt",
    Path(__file__).with_name("glossaries") / "core_theravada_terms.txt",
    Path(__file__).with_name("glossaries") / "lineages" / "ajahn_chah.txt",
    Path(__file__).with_name("glossaries") / "local_teachers_and_places.txt",
]
FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
            },
        }
    },
    "required": ["items"],
}


CORRECTION_GUIDANCE = """\
You are correcting subtitles for Theravada Buddhist meditation talks.

Goals:
- Fix obvious ASR errors in English.
- Correct Buddhist, Pali, and Thai Forest terms when the intended term is clear.
- Correct chant formulas when the transcript is a clear phonetic corruption of a known chant.
- Keep the speaker's meaning and tone.
- Keep each output item aligned to the same numbered input item.

Important Buddhist corrections:
- "Namo tassa bhagavato arahato samma-sambuddhassa"
- "Buddham saranam gacchami"
- "Dhammam saranam gacchami"
- "Sangham saranam gacchami"
- "Ajahn Chah"
- "Ajahn Mahabua"
- "Ajahn"
- "Theravada"
- "Dhamma"
- "Sangha"
- "Bodhi tree"
- "samsara"
- "anumodana"
- "Mahamangala Sutta"
- "sadhu"
- "mettā" may be written as "metta"

Rules:
- Do not rewrite timestamps or numbering.
- Do not summarize.
- Do not add missing paragraphs or commentary.
- If a line is too uncertain, make only conservative fixes.
- Prefer plain ASCII transliteration like "metta", "samsara", "anumodana" unless the input already uses diacritics.
- Use the provided glossary as a correction hint set, especially for chants, lineage terms, and proper names.
- Return only JSON with this shape:
  {"items":[{"id":1,"text":"corrected text"}]}
"""


@dataclass
class Cue:
    index: int
    timing: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Naive local LLM correction pass for VTT subtitles.")
    parser.add_argument("input_path", type=str, help="Source VTT file.")
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="Where to write the corrected VTT. Defaults to same name with .qwen.vtt",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL, help="Ollama generate endpoint.")
    parser.add_argument(
        "--glossary-file",
        action="append",
        default=[],
        help="Additional glossary file to append. Can be passed more than once.",
    )
    parser.add_argument(
        "--no-default-glossaries",
        action="store_true",
        help="Do not load the built-in glossary stack from the project.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of cues per request. 0 means send the whole transcript in one request.",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--context-window", type=int, default=8192, help="num_ctx for Ollama.")
    parser.add_argument(
        "--keep-model-loaded",
        action="store_true",
        help="Do not unload the Ollama model after the correction run finishes.",
    )
    return parser.parse_args()


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


def estimate_tokens(text: str) -> int:
    # Conservative English-ish heuristic for Ollama prompt sizing.
    return max(1, math.ceil(len(text) / 4))


def build_cue_batches(
    cues: list[Cue],
    batch_size: int,
    max_batch_characters: int,
    *,
    glossary_lines: list[str],
    max_prompt_tokens: int,
) -> list[list[Cue]]:
    if not cues:
        return []
    if (
        (batch_size <= 0 or batch_size >= len(cues))
        and max_batch_characters <= 0
        and max_prompt_tokens <= 0
    ):
        return [cues]

    batches: list[list[Cue]] = []
    current_batch: list[Cue] = []
    current_characters = 0

    for cue in cues:
        cue_characters = len(cue.text)
        would_hit_count_cap = batch_size > 0 and len(current_batch) >= batch_size
        would_hit_character_cap = (
            max_batch_characters > 0
            and current_batch
            and current_characters + cue_characters > max_batch_characters
        )
        would_hit_prompt_cap = False
        if max_prompt_tokens > 0 and current_batch:
            candidate_prompt = build_prompt(current_batch + [cue], glossary_lines)
            would_hit_prompt_cap = estimate_tokens(candidate_prompt) > max_prompt_tokens
        if would_hit_count_cap or would_hit_character_cap:
            batches.append(current_batch)
            current_batch = []
            current_characters = 0
        elif would_hit_prompt_cap:
            batches.append(current_batch)
            current_batch = []
            current_characters = 0

        current_batch.append(cue)
        current_characters += cue_characters

    if current_batch:
        batches.append(current_batch)
    return batches


def load_glossary_lines(glossary_paths: list[Path]) -> list[str]:
    lines: list[str] = []
    for path in glossary_paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


def build_prompt(batch: list[Cue], glossary_lines: list[str]) -> str:
    items = [{"id": cue.index, "text": cue.text} for cue in batch]
    payload = json.dumps({"items": items}, ensure_ascii=False, indent=2)
    glossary_block = "\n".join(f"- {line}" for line in glossary_lines) if glossary_lines else "- (none)"
    return textwrap.dedent(
        f"""\
        {CORRECTION_GUIDANCE}

        Glossary and correction hints:
        {glossary_block}

        Here are the subtitle items to correct:
        {payload}
        """
    )


def call_ollama(
    url: str,
    model: str,
    prompt: str,
    temperature: float,
    context_window: int,
    format_schema: dict[str, object] | None = None,
) -> dict[str, object]:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": format_schema or FORMAT_SCHEMA,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_ctx": context_window,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach Ollama at {url}: {exc}") from exc

    raw_response = result.get("response")
    if (not isinstance(raw_response, str) or not raw_response.strip()) and isinstance(result.get("thinking"), str):
        raw_response = result.get("thinking")
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise RuntimeError("Ollama returned an empty response.")

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama did not return valid JSON: {raw_response[:400]}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("Ollama returned JSON, but not an object.")
    return parsed


def extract_batch_corrections(response_json: dict[str, object], batch: list[Cue]) -> dict[int, str]:
    raw_items = response_json.get("items")
    if not isinstance(raw_items, list):
        raise RuntimeError("Model response did not contain an 'items' list.")

    corrections: dict[int, str] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        text = item.get("text")
        if isinstance(item_id, int) and isinstance(text, str):
            corrections[item_id] = text.strip()

    missing = [cue.index for cue in batch if cue.index not in corrections]
    if missing:
        raise RuntimeError(f"Model response missed cue ids: {missing}")

    return corrections


def process_batch(
    batch: list[Cue],
    glossary_lines: list[str],
    url: str,
    model: str,
    temperature: float,
    context_window: int,
) -> dict[int, str]:
    prompt = build_prompt(batch, glossary_lines)
    try:
        response_json = call_ollama(
            url=url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            context_window=context_window,
        )
        return extract_batch_corrections(response_json, batch)
    except Exception:
        if len(batch) == 1:
            raise
        midpoint = len(batch) // 2
        left = process_batch(batch[:midpoint], glossary_lines, url, model, temperature, context_window)
        right = process_batch(batch[midpoint:], glossary_lines, url, model, temperature, context_window)
        left.update(right)
        return left


def render_vtt(cues: list[Cue], corrected_text: dict[int, str]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for cue in cues:
        lines.append(str(cue.index))
        lines.append(cue.timing)
        lines.append(corrected_text.get(cue.index, cue.text))
        lines.append("")
    return "\n".join(lines)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.qwen.vtt")


def find_ollama_executable() -> str | None:
    executable = shutil.which("ollama")
    if executable:
        return executable

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None

    candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
    if candidate.exists():
        return str(candidate)
    return None


def unload_ollama_model(model: str, ollama_url: str) -> None:
    executable = find_ollama_executable()
    if executable:
        subprocess.run(
            [executable, "stop", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return

    request = urllib.request.Request(
        ollama_url,
        data=json.dumps(
            {
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60):
            return
    except Exception:
        return


def correct_cues(
    cues: list[Cue],
    *,
    glossary_paths: list[Path] | None = None,
    glossary_lines: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temperature: float = 0.1,
    context_window: int = 8192,
    progress: bool = False,
) -> dict[int, str]:
    if not cues:
        raise ValueError("No VTT cues were provided for correction.")

    loaded_glossary_lines = glossary_lines
    if loaded_glossary_lines is None:
        loaded_glossary_lines = load_glossary_lines(glossary_paths or [])

    corrected: dict[int, str] = {}
    resolved_batch_size = batch_size
    max_batch_characters = DEFAULT_MAX_BATCH_CHARACTERS
    max_prompt_tokens = DEFAULT_MAX_PROMPT_TOKENS
    batches = build_cue_batches(
        cues,
        resolved_batch_size,
        max_batch_characters,
        glossary_lines=loaded_glossary_lines,
        max_prompt_tokens=max_prompt_tokens,
    )
    if progress and glossary_paths:
        print("Glossaries:", flush=True)
        for path in glossary_paths:
            print(f"  {path}", flush=True)
    if progress:
        print(f"Glossary entries loaded: {len(loaded_glossary_lines)}", flush=True)
        label = "whole transcript" if resolved_batch_size <= 0 else str(resolved_batch_size)
        print(f"Cleanup batch size: {label}", flush=True)
        character_label = "unlimited" if max_batch_characters <= 0 else str(max_batch_characters)
        print(f"Cleanup max batch characters: {character_label}", flush=True)
        token_label = "unlimited" if max_prompt_tokens <= 0 else str(max_prompt_tokens)
        print(f"Cleanup max prompt tokens: {token_label}", flush=True)
    for batch_number, batch in enumerate(batches, start=1):
        if progress:
            batch_characters = sum(len(cue.text) for cue in batch)
            prompt_tokens = estimate_tokens(build_prompt(batch, loaded_glossary_lines))
            print(
                " ".join(
                    [
                        f"Batch {batch_number}/{len(batches)}:",
                        f"cues {batch[0].index}-{batch[-1].index}",
                        f"chars={batch_characters}",
                        f"prompt_tokens~={prompt_tokens}",
                    ]
                ),
                flush=True,
            )
        corrected.update(
            process_batch(
                batch=batch,
                glossary_lines=loaded_glossary_lines,
                url=ollama_url,
                model=model,
                temperature=temperature,
                context_window=context_window,
            )
        )
    return corrected


def correct_vtt_file(
    input_path: Path,
    *,
    output_path: Path | None = None,
    glossary_paths: list[Path] | None = None,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temperature: float = 0.1,
    context_window: int = 8192,
    progress: bool = False,
) -> Path:
    cues = parse_vtt(input_path)
    if not cues:
        raise ValueError(f"No VTT cues found in {input_path}")

    resolved_output_path = output_path or default_output_path(input_path)
    loaded_glossary_lines = load_glossary_lines(glossary_paths or [])
    corrected = correct_cues(
        cues,
        glossary_paths=glossary_paths or [],
        glossary_lines=loaded_glossary_lines,
        model=model,
        ollama_url=ollama_url,
        batch_size=batch_size,
        temperature=temperature,
        context_window=context_window,
        progress=progress,
    )
    resolved_output_path.write_text(render_vtt(cues, corrected), encoding="utf-8", newline="\n")
    return resolved_output_path


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else default_output_path(input_path)
    )
    glossary_paths = [] if args.no_default_glossaries else list(DEFAULT_GLOSSARY_FILES)
    glossary_paths.extend(Path(path).expanduser().resolve() for path in args.glossary_file)
    try:
        written_path = correct_vtt_file(
            input_path,
            output_path=output_path,
            glossary_paths=glossary_paths,
            model=args.model,
            ollama_url=args.ollama_url,
            batch_size=args.batch_size,
            temperature=args.temperature,
            context_window=args.context_window,
            progress=True,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Wrote corrected VTT: {written_path}", flush=True)
    if not args.keep_model_loaded:
        unload_ollama_model(args.model, args.ollama_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
