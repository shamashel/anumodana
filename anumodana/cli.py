"""Click CLI — the user-facing entry point for Anumodana."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from click_default_group import DefaultGroup

from anumodana.helpers.config import AnumodanaConfig, load_config, resolve_runtime_config

logger = logging.getLogger("anumodana")


def _setup_logging(verbose: bool = False) -> None:
    """Configure the ``anumodana`` logger for console output."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger("anumodana")
    root_logger.setLevel(level)
    # Avoid duplicate handlers on repeated calls.
    if not root_logger.handlers:
        root_logger.addHandler(handler)


# ---------------------------------------------------------------------------
# CLI group — bare ``anumodana`` runs ``all`` by default.
# ---------------------------------------------------------------------------
@click.group(cls=DefaultGroup, default="pipeline", default_if_no_args=False)
@click.version_option(version="0.1.0", prog_name="anumodana")
def cli():
    """Anumodana — transcription pipeline for Theravada Buddhist talks."""


# ---------------------------------------------------------------------------
# pipeline — full pipeline
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--root", default=None, type=click.Path(exists=True), help="Root directory to scan.")
@click.option("--overwrite", is_flag=True, help="Rebuild existing outputs.")
@click.option("--limit", default=0, type=int, help="Process only the first N jobs.")
@click.option("--dry-run", is_flag=True, help="List what would run without doing it.")
@click.option("--local", is_flag=True, help="Run fixer and review locally instead of via Ollama Cloud.")
@click.option("--review", is_flag=True, help="Perform the AI review pass and generate a manifest.")
@click.option("--glossary-file", multiple=True, help="Additional glossary file(s).")
@click.option("--no-default-glossaries", is_flag=True, help="Do not load the built-in glossary stack.")
@click.option("--manifest-path", default="", help="Where to write the review manifest CSV.")
@click.option("--keep-models-loaded", is_flag=True, help="Do not unload models after the run.")
@click.option("--verbose", is_flag=True, help="Show verbose library diagnostics.")
def pipeline(
    root, overwrite, limit, dry_run, local, review,
    glossary_file, no_default_glossaries, manifest_path, keep_models_loaded, verbose,
):
    """Run the transcription pipeline: transcribe → fix (optional review)."""
    _setup_logging(verbose)
    from anumodana.commands.pipeline import DEFAULT_ROOT, run_pipeline

    config = resolve_runtime_config(load_config(), local=local)
    resolved_root = Path(root) if root else DEFAULT_ROOT

    if not config.api_key and config.mode == "cloud":
        logger.info("")
        logger.info("No API key found. Falling back to local Ollama.")
        logger.info("To use Ollama Cloud (free, no GPU needed for fixer/review):")
        logger.info("  Run: anumodana onboard")
        logger.info("")
        config = resolve_runtime_config(config, local=True)

    raise SystemExit(run_pipeline(
        config=config,
        root=resolved_root,
        overwrite=overwrite,
        limit=limit,
        dry_run=dry_run,
        review=review,
        glossary_extra=list(glossary_file),
        no_default_glossaries=no_default_glossaries,
        manifest_path_arg=manifest_path,
        keep_models_loaded=keep_models_loaded,
        verbose=verbose,
    ))


# ---------------------------------------------------------------------------
# transcribe — ASR only (no fixer, no review)
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--root", default=None, type=click.Path(exists=True), help="Root directory to scan.")
@click.option("--overwrite", is_flag=True, help="Rebuild existing outputs.")
@click.option("--limit", default=0, type=int, help="Process only the first N jobs.")
@click.option("--dry-run", is_flag=True, help="List what would run without doing it.")
@click.option("--keep-models-loaded", is_flag=True, help="Do not unload models after the run.")
@click.option("--verbose", is_flag=True, help="Show verbose library diagnostics.")
def transcribe(root, overwrite, limit, dry_run, keep_models_loaded, verbose):
    """Transcribe audio/video files to raw VTT using Parakeet (no fixer, no review)."""
    _setup_logging(verbose)
    from anumodana.commands.pipeline import DEFAULT_ROOT
    from anumodana.commands.transcribe import run_transcribe

    config = load_config()
    resolved_root = Path(root) if root else DEFAULT_ROOT

    raise SystemExit(run_transcribe(
        config=config,
        root=resolved_root,
        overwrite=overwrite,
        limit=limit,
        dry_run=dry_run,
        keep_models_loaded=keep_models_loaded,
        verbose=verbose,
    ))


# ---------------------------------------------------------------------------
# fix — standalone fixer pass on an existing VTT
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output-path", default="", help="Where to write the corrected VTT.")
@click.option("--local", is_flag=True, help="Run the fixer locally instead of via Ollama Cloud.")
@click.option("--glossary-file", multiple=True, help="Additional glossary file(s).")
@click.option("--no-default-glossaries", is_flag=True, help="Do not load the built-in glossary stack.")
@click.option("--keep-model-loaded", is_flag=True, help="Do not unload the model after the run.")
@click.option("--verbose", is_flag=True, help="Show verbose diagnostics.")
def fix(input_path, output_path, local, glossary_file, no_default_glossaries, keep_model_loaded, verbose):
    """Run AI correction on a raw VTT file."""
    _setup_logging(verbose)
    from anumodana.commands.fix import fix_vtt_file
    from anumodana.helpers.glossary import build_glossary_paths
    from anumodana.helpers.models import fixer_output_path
    from anumodana.helpers.ollama import build_client, unload_model

    config = resolve_runtime_config(load_config(), local=local)
    fixer_config = config.fixer

    if not config.api_key and config.mode == "cloud":
        logger.info("No API key found. Falling back to local Ollama.")
        config = resolve_runtime_config(config, local=True)
        fixer_config = config.fixer

    client = build_client(fixer_config.ollama, api_key=config.api_key)
    resolved_input = Path(input_path).expanduser().resolve()
    resolved_output = Path(output_path).expanduser().resolve() if output_path else fixer_output_path(resolved_input)

    glossary_paths = build_glossary_paths(
        list(glossary_file),
        include_defaults=not no_default_glossaries,
    )

    try:
        written = fix_vtt_file(
            resolved_input,
            client=client,
            config=fixer_config.ollama,
            output_path=resolved_output,
            glossary_paths=glossary_paths,
            batch_size=fixer_config.batch_size,
            max_batch_characters=fixer_config.max_batch_characters,
            max_prompt_tokens=fixer_config.max_prompt_tokens,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1)

    logger.info("Wrote corrected VTT: %s", written)
    if not keep_model_loaded:
        unload_model(client, fixer_config.ollama)


# ---------------------------------------------------------------------------
# review — standalone review pass
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("raw_vtt_path", type=click.Path(exists=True))
@click.argument("cleaned_vtt_path", type=click.Path(exists=True))
@click.option("--output-json", default="", help="Write structured review JSON here.")
@click.option("--output-md", default="", help="Write human-readable review markdown here.")
@click.option("--local", is_flag=True, help="Run the review locally instead of via Ollama Cloud.")
@click.option("--glossary-file", multiple=True, help="Additional glossary file(s).")
@click.option("--no-default-glossaries", is_flag=True, help="Do not load the built-in glossary stack.")
@click.option("--keep-model-loaded", is_flag=True, help="Do not unload the model after the run.")
@click.option("--verbose", is_flag=True, help="Show verbose diagnostics.")
def review(
    raw_vtt_path, cleaned_vtt_path, output_json, output_md,
    local, glossary_file, no_default_glossaries, keep_model_loaded, verbose,
):
    """Review a cleaned VTT against its raw transcript."""
    _setup_logging(verbose)
    import json as json_mod

    from anumodana.helpers.glossary import build_glossary_paths
    from anumodana.helpers.models import review_json_standalone_path, review_md_standalone_path
    from anumodana.helpers.ollama import build_client, unload_model
    from anumodana.commands.review import render_review_markdown
    from anumodana.commands.review import review_transcripts as _review_transcripts

    config = resolve_runtime_config(load_config(), local=local)
    review_config = config.review

    if not config.api_key and config.mode == "cloud":
        logger.info("No API key found. Falling back to local Ollama.")
        config = resolve_runtime_config(config, local=True)
        review_config = config.review

    client = build_client(review_config.ollama, api_key=config.api_key)
    raw_path = Path(raw_vtt_path).expanduser().resolve()
    cleaned_path = Path(cleaned_vtt_path).expanduser().resolve()
    json_out = Path(output_json).expanduser().resolve() if output_json else review_json_standalone_path(cleaned_path)
    md_out = Path(output_md).expanduser().resolve() if output_md else review_md_standalone_path(cleaned_path)

    glossary_paths = build_glossary_paths(
        list(glossary_file),
        include_defaults=not no_default_glossaries,
    )

    try:
        review_result = _review_transcripts(
            raw_vtt_path=raw_path,
            cleaned_vtt_path=cleaned_path,
            client=client,
            config=review_config.ollama,
            glossary_paths=glossary_paths,
        )
    finally:
        if not keep_model_loaded:
            unload_model(client, review_config.ollama)

    json_out.write_text(
        json_mod.dumps(review_result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8", newline="\n",
    )
    md_out.write_text(
        render_review_markdown(review_result, raw_vtt_path=raw_path, cleaned_vtt_path=cleaned_path),
        encoding="utf-8", newline="\n",
    )
    logger.info("Wrote review JSON: %s", json_out)
    logger.info("Wrote review markdown: %s", md_out)


# ---------------------------------------------------------------------------
# onboard — interactive setup wizard
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--mode", type=click.Choice(["cloud", "local"]), help="Onboarding mode.")
@click.option("--api-key", default="", help="Ollama Cloud API key.")
@click.option("--yes", is_flag=True, help="Auto-confirm all prompts.")
def onboard(mode, api_key, yes):
    """Interactive setup wizard — configure Anumodana for first use."""
    _setup_logging()
    from anumodana.commands.onboard import run_onboard
    run_onboard(mode=mode, api_key=api_key, yes=yes)
