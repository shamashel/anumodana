from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from anumodana.glossary import DEFAULT_GLOSSARY_FILES, build_glossary_paths, load_glossary_lines
from anumodana.manifest import (
    REVIEW_MANIFEST_FIELDNAMES,
    append_manifest_row,
    build_review_manifest_row,
)
from anumodana.ollama import DEFAULT_MODEL, DEFAULT_OLLAMA_URL, call_ollama, unload_ollama_model
from anumodana.output_paths import review_json_output_path, review_md_output_path


DEFAULT_CONTEXT_WINDOW = 32768
DEFAULT_TEMPERATURE = 0.1

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "review_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timing": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["timing", "note"],
            },
        },
        "concerns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timing": {"type": "string"},
                    "issue_type": {"type": "string"},
                    "severity": {"type": "string"},
                    "why_weird": {"type": "string"},
                    "suggested_action": {"type": "string"},
                },
                "required": ["timing", "issue_type", "severity", "why_weird", "suggested_action"],
            },
        },
        "needs_human_review": {"type": "boolean"},
    },
    "required": ["summary", "review_notes", "concerns", "needs_human_review"],
}

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
- `review_notes` should be sparse and useful, not chain-of-thought.
- `concerns` should be specific and reference exact cue timings.
- `needs_human_review` should be true if any concern is materially risky, semantically broken, or likely misleading.

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


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Review a cleaned VTT against its raw ASR transcript and emit structured review artifacts."
    )
    parser.add_argument("raw_vtt_path", type=str, help="Raw ASR VTT path.")
    parser.add_argument("cleaned_vtt_path", type=str, help="Cleaned VTT path.")
    parser.add_argument("--output-json", type=str, default="", help="Write structured review JSON here.")
    parser.add_argument("--output-md", type=str, default="", help="Write a human-readable review note here.")
    parser.add_argument("--output-csv", type=str, default="", help="Optionally append a one-row summary CSV here.")
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
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW, help="num_ctx for Ollama.")
    parser.add_argument(
        "--keep-model-loaded",
        action="store_true",
        help="Do not unload the Ollama model after the review run finishes.",
    )
    return parser.parse_args(argv)


def default_json_output_path(cleaned_vtt_path: Path) -> Path:
    return review_json_output_path(cleaned_vtt_path)


def default_md_output_path(cleaned_vtt_path: Path) -> Path:
    return review_md_output_path(cleaned_vtt_path)


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
    glossary_paths: list[Path] | None = None,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> dict[str, object]:
    raw_vtt = raw_vtt_path.read_text(encoding="utf-8")
    cleaned_vtt = cleaned_vtt_path.read_text(encoding="utf-8")
    glossary_lines = load_glossary_lines(glossary_paths or [])
    prompt = build_review_prompt(raw_vtt, cleaned_vtt, glossary_lines)
    result = call_ollama(
        url=ollama_url,
        model=model,
        prompt=prompt,
        temperature=temperature,
        context_window=context_window,
        format_schema=REVIEW_SCHEMA,
    )

    summary = result.get("summary")
    review_notes = result.get("review_notes")
    concerns = result.get("concerns")
    needs_human_review = result.get("needs_human_review")
    if not isinstance(summary, str):
        raise RuntimeError("Review output did not include a summary string.")
    if not isinstance(review_notes, list):
        raise RuntimeError("Review output did not include a review_notes list.")
    if not isinstance(concerns, list):
        raise RuntimeError("Review output did not include a concerns list.")
    if not isinstance(needs_human_review, bool):
        raise RuntimeError("Review output did not include a needs_human_review boolean.")
    return result


def render_review_markdown(
    review: dict[str, object],
    *,
    raw_vtt_path: Path,
    cleaned_vtt_path: Path,
) -> str:
    lines: list[str] = ["# Review", ""]
    lines.append(f"Raw transcript: {raw_vtt_path}")
    lines.append(f"Cleaned transcript: {cleaned_vtt_path}")
    lines.append("")
    lines.append(f"Needs human review: {'yes' if review['needs_human_review'] else 'no'}")
    lines.append("")
    lines.append("## Summary")
    lines.append(str(review["summary"]).strip() or "(none)")
    lines.append("")

    review_notes = review.get("review_notes", [])
    lines.append("## Review Notes")
    if isinstance(review_notes, list) and review_notes:
        for note in review_notes:
            if not isinstance(note, dict):
                continue
            timing = str(note.get("timing", "")).strip() or "(no timing)"
            text = str(note.get("note", "")).strip()
            if text:
                lines.append(f"- {timing}: {text}")
    else:
        lines.append("- None.")
    lines.append("")

    concerns = review.get("concerns", [])
    lines.append("## Concerns")
    if isinstance(concerns, list) and concerns:
        for concern in concerns:
            if not isinstance(concern, dict):
                continue
            timing = str(concern.get("timing", "")).strip() or "(no timing)"
            issue_type = str(concern.get("issue_type", "")).strip() or "unspecified"
            severity = str(concern.get("severity", "")).strip() or "unspecified"
            why_weird = str(concern.get("why_weird", "")).strip()
            suggested_action = str(concern.get("suggested_action", "")).strip()
            lines.append(f"- {timing} [{severity}] {issue_type}")
            if why_weird:
                lines.append(f"  Why: {why_weird}")
            if suggested_action:
                lines.append(f"  Action: {suggested_action}")
    else:
        lines.append("- None.")

    return "\n".join(lines) + "\n"


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
) -> int:
    args = parse_args(argv, prog=prog)
    raw_vtt_path = Path(args.raw_vtt_path).expanduser().resolve()
    cleaned_vtt_path = Path(args.cleaned_vtt_path).expanduser().resolve()
    if not raw_vtt_path.exists():
        print(f"Raw VTT does not exist: {raw_vtt_path}", file=sys.stderr)
        return 1
    if not cleaned_vtt_path.exists():
        print(f"Cleaned VTT does not exist: {cleaned_vtt_path}", file=sys.stderr)
        return 1

    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else default_json_output_path(cleaned_vtt_path)
    output_md = Path(args.output_md).expanduser().resolve() if args.output_md else default_md_output_path(cleaned_vtt_path)

    glossary_paths = build_glossary_paths(
        args.glossary_file,
        include_defaults=not args.no_default_glossaries,
    )

    try:
        review = review_transcripts(
            raw_vtt_path=raw_vtt_path,
            cleaned_vtt_path=cleaned_vtt_path,
            glossary_paths=glossary_paths,
            model=args.model,
            ollama_url=args.ollama_url,
            temperature=args.temperature,
            context_window=args.context_window,
        )
    finally:
        if not args.keep_model_loaded:
            unload_ollama_model(args.model, args.ollama_url)

    output_json.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    output_md.write_text(
        render_review_markdown(review, raw_vtt_path=raw_vtt_path, cleaned_vtt_path=cleaned_vtt_path),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote review JSON: {output_json}", flush=True)
    print(f"Wrote review markdown: {output_md}", flush=True)

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
        append_manifest_row(
            output_csv,
            build_review_manifest_row(
                raw_vtt_path=raw_vtt_path,
                cleaned_vtt_path=cleaned_vtt_path,
                review_json_path=output_json,
                review_md_path=output_md,
                review=review,
            ),
            fieldnames=REVIEW_MANIFEST_FIELDNAMES,
        )
        print(f"Updated review CSV: {output_csv}", flush=True)

    return 0
