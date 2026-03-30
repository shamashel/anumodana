"""AI review pass — compares raw and cleaned transcripts for quality concerns."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

from ollama import Client

from anumodana.helpers.config import OllamaConfig
from anumodana.helpers.glossary import load_glossary_lines
from anumodana.helpers.models import ReviewResponse
from anumodana.helpers.ollama import call_ollama

logger = logging.getLogger("anumodana")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
REVIEW_GUIDANCE = """\
You are reviewing cleaned subtitles for Theravada Buddhist meditation talks.

You will receive:
- a raw transcript produced by ASR
- a cleaned transcript produced by a local correction pass

Your tasks:
- review the cleaned transcript against the raw transcript
- identify noteworthy cleanup choices
- identify places that still look suspicious or likely need human attention
- decide whether a human should review this file

Output rules:
- Return one JSON object only.
- `summary` should be short and practical.
- `review_notes` should be sparse and useful, not chain-of-thought. Use `timing` not `cue_id`.
- `concerns` should be specific and reference exact cue timings in the `timing` field.
- `needs_human_review` should be true if any concern is materially risky, semantically broken, or likely misleading.

Example JSON output:
{
  "summary": "The transcript is mostly clean, except for a few Pali terms and a corrupted chant at the beginning.",
  "review_notes": [
    { "timing": "00:00:15,000", "note": "Corrected 'Ajahn Brahm' capitalization." }
  ],
  "concerns": [
    {
      "timing": "00:01:45,000",
      "issue_type": "chant",
      "severity": "high",
      "why_weird": "The opening chant was severely mangled by ASR and the fixer could not recover it.",
      "suggested_action": "Manually re-transcribe the chant."
    }
  ],
  "needs_human_review": true
}

Focus on:
- corrupted chants or refuge formulas
- Buddhist / Pali / Thai Forest names and terms
- semantically broken sentences
- obvious non-speech garbage
- places where the cleaned transcript may still be overconfident

Do not:
- invent timestamps
- rewrite the transcript
- provide hidden reasoning or verbose deliberation
- flag ordinary punctuation cleanup as a major concern
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_review_prompt(raw_vtt: str, cleaned_vtt: str, glossary_lines: list[str]) -> str:
    glossary_block = "\n".join(f"- {line}" for line in glossary_lines) if glossary_lines else "- (none)"
    return textwrap.dedent(
        f"""\
        {REVIEW_GUIDANCE}

        Glossary and correction hints:
        {glossary_block}

        Raw WEBVTT:
        ```vtt
        {raw_vtt}
        ```

        Cleaned WEBVTT:
        ```vtt
        {cleaned_vtt}
        ```
        """
    )


def review_transcripts(
    raw_vtt_path: Path,
    cleaned_vtt_path: Path,
    *,
    client: Client,
    config: OllamaConfig,
    glossary_paths: list[Path] | None = None,
) -> ReviewResponse:
    """Run the review pass, returning a validated :class:`ReviewResponse`."""
    raw_vtt = raw_vtt_path.read_text(encoding="utf-8")
    cleaned_vtt = cleaned_vtt_path.read_text(encoding="utf-8")
    glossary_lines = load_glossary_lines(glossary_paths or [])
    prompt = build_review_prompt(raw_vtt, cleaned_vtt, glossary_lines)
    raw = call_ollama(
        client=client,
        config=config,
        prompt=prompt,
        format_schema=ReviewResponse.model_json_schema(),
    )
    return ReviewResponse.model_validate(raw)


def render_review_markdown(
    review: ReviewResponse,
    *,
    raw_vtt_path: Path,
    cleaned_vtt_path: Path,
) -> str:
    """Render a human-readable review markdown document."""
    lines: list[str] = ["# Review", ""]
    lines.append(f"Raw transcript: {raw_vtt_path}")
    lines.append(f"Cleaned transcript: {cleaned_vtt_path}")
    lines.append("")
    lines.append(f"Needs human review: {'yes' if review.needs_human_review else 'no'}")
    lines.append("")
    lines.append("## Summary")
    lines.append(review.summary.strip() or "(none)")
    lines.append("")

    lines.append("## Review Notes")
    if review.review_notes:
        for note in review.review_notes:
            timing = note.timing.strip() or "(no timing)"
            text = note.note.strip()
            if text:
                lines.append(f"- {timing}: {text}")
    else:
        lines.append("- None.")
    lines.append("")

    lines.append("## Concerns")
    if review.concerns:
        for concern in review.concerns:
            timing = concern.timing.strip() or "(no timing)"
            issue_type = concern.issue_type.strip() or "unspecified"
            severity = concern.severity.strip() or "unspecified"
            lines.append(f"- {timing} [{severity}] {issue_type}")
            if concern.why_weird.strip():
                lines.append(f"  Why: {concern.why_weird.strip()}")
            if concern.suggested_action.strip():
                lines.append(f"  Action: {concern.suggested_action.strip()}")
    else:
        lines.append("- None.")

    return "\n".join(lines) + "\n"
