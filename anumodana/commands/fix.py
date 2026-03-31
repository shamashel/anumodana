"""AI fixer pass — corrects ASR subtitle errors using Ollama."""

from __future__ import annotations

import json
import logging
import math
import textwrap
from pathlib import Path

from ollama import Client

from anumodana.helpers.config import OllamaConfig
from anumodana.helpers.glossary import load_glossary_lines
from anumodana.helpers.models import CorrectionResponse, fixer_output_path
from anumodana.helpers.ollama import call_ollama
from anumodana.helpers.transcript import Cue, parse_vtt, render_vtt

logger = logging.getLogger("anumodana")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
FIXER_GUIDANCE = """\
You are correcting subtitles for Theravada Buddhist meditation talks.

Goals:
- Fix obvious ASR errors in English.
- Correct Buddhist, Pali, and Thai Forest terms when the intended term is clear.
- Correct chant formulas when the transcript is a clear phonetic corruption of a known chant.
- Keep the speaker's meaning and tone.
- Keep each output item aligned to the same numbered input item.

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Conservative English-ish heuristic for Ollama prompt sizing."""
    return max(1, math.ceil(len(text) / 4))


def build_prompt(batch: list[Cue], glossary_lines: list[str]) -> str:
    items = [{"id": cue.index, "text": cue.text} for cue in batch]
    payload = json.dumps({"items": items}, ensure_ascii=False, indent=2)
    glossary_block = "\n".join(f"- {line}" for line in glossary_lines) if glossary_lines else "- (none)"
    return textwrap.dedent(
        f"""\
        {FIXER_GUIDANCE}

        Glossary and correction hints:
        {glossary_block}

        Here are the subtitle items to correct:
        {payload}
        """
    )


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
        if would_hit_count_cap or would_hit_character_cap or would_hit_prompt_cap:
            batches.append(current_batch)
            current_batch = []
            current_characters = 0

        current_batch.append(cue)
        current_characters += cue_characters

    if current_batch:
        batches.append(current_batch)
    return batches


def extract_batch_corrections(response: CorrectionResponse, batch: list[Cue]) -> dict[int, str]:
    corrections: dict[int, str] = {}
    for item in response.items:
        corrections[item.id] = item.text.strip()

    missing = [cue.index for cue in batch if cue.index not in corrections]
    if missing:
        raise RuntimeError(f"Model response missed cue ids: {missing}")
    return corrections


def process_batch(
    batch: list[Cue],
    glossary_lines: list[str],
    client: Client,
    config: OllamaConfig,
) -> dict[int, str]:
    prompt = build_prompt(batch, glossary_lines)
    try:
        raw = call_ollama(
            client=client,
            config=config,
            prompt=prompt,
            format_schema=CorrectionResponse.model_json_schema(),
        )
        response = CorrectionResponse.model_validate(raw)
        return extract_batch_corrections(response, batch)
    except Exception:
        if len(batch) == 1:
            raise
        midpoint = len(batch) // 2
        left = process_batch(batch[:midpoint], glossary_lines, client, config)
        right = process_batch(batch[midpoint:], glossary_lines, client, config)
        left.update(right)
        return left


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fix_cues(
    cues: list[Cue],
    *,
    client: Client,
    config: OllamaConfig,
    glossary_paths: list[Path] | None = None,
    batch_size: int = 16,
    max_batch_characters: int = 3200,
    max_prompt_tokens: int = 2600,
) -> dict[int, str]:
    """Run the fixer pass over a list of VTT cues, returning corrections."""
    if not cues:
        raise ValueError("No VTT cues were provided for correction.")

    glossary_lines = load_glossary_lines(glossary_paths or [])

    corrected: dict[int, str] = {}
    batches = build_cue_batches(
        cues,
        batch_size,
        max_batch_characters,
        glossary_lines=glossary_lines,
        max_prompt_tokens=max_prompt_tokens,
    )
    if glossary_paths:
        logger.info("Glossaries:")
        for path in glossary_paths:
            logger.info("  %s", path)
    logger.info("Glossary entries loaded: %d", len(glossary_lines))
    label = "whole transcript" if batch_size <= 0 else str(batch_size)
    logger.info("Fixer batch size: %s", label)
    character_label = "unlimited" if max_batch_characters <= 0 else str(max_batch_characters)
    logger.info("Fixer max batch characters: %s", character_label)
    token_label = "unlimited" if max_prompt_tokens <= 0 else str(max_prompt_tokens)
    logger.info("Fixer max prompt tokens: %s", token_label)

    for batch_number, batch in enumerate(batches, start=1):
        batch_characters = sum(len(cue.text) for cue in batch)
        prompt_tokens = estimate_tokens(build_prompt(batch, glossary_lines))
        logger.info(
            "Batch %d/%d: cues %d-%d chars=%d prompt_tokens~=%d",
            batch_number, len(batches),
            batch[0].index, batch[-1].index,
            batch_characters, prompt_tokens,
        )
        corrected.update(
            process_batch(
                batch=batch,
                glossary_lines=glossary_lines,
                client=client,
                config=config,
            )
        )
    return corrected


def fix_vtt_file(
    input_path: Path,
    *,
    client: Client,
    config: OllamaConfig,
    output_path: Path | None = None,
    glossary_paths: list[Path] | None = None,
    batch_size: int = 16,
    max_batch_characters: int = 3200,
    max_prompt_tokens: int = 2600,
) -> Path:
    """Run the fixer pass on a VTT file and write the corrected output."""
    cues = parse_vtt(input_path)
    if not cues:
        raise ValueError(f"No VTT cues found in {input_path}")

    resolved_output_path = output_path or fixer_output_path(input_path)
    corrected = fix_cues(
        cues,
        client=client,
        config=config,
        glossary_paths=glossary_paths or [],
        batch_size=batch_size,
        max_batch_characters=max_batch_characters,
        max_prompt_tokens=max_prompt_tokens,
    )
    resolved_output_path.write_text(render_vtt(cues, corrected), encoding="utf-8", newline="\n")
    return resolved_output_path
