"""Interactive onboarding wizard for first-time Anumodana setup."""

from __future__ import annotations

import logging
import shutil

import click

from anumodana.helpers.config import (
    CLOUD_OLLAMA_HOST,
    CONFIG_PATH,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_LOCAL_MODEL,
    AnumodanaConfig,
    OllamaConfig,
    load_config,
    save_config,
)

logger = logging.getLogger("anumodana")


def _check_ffmpeg() -> bool:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        click.echo("  ✓ FFmpeg found")
        return True
    click.echo("  ✗ FFmpeg not found — install it or add it to PATH.")
    click.echo("    https://ffmpeg.org/download.html")
    return False


def _check_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            click.echo(f"  ✓ GPU detected: {name}")
            click.echo("    Transcription will run on GPU (faster).")
        else:
            click.echo("  ℹ No GPU detected — transcription will use CPU (this is fine, just slower).")
    except ImportError:
        click.echo("  ℹ PyTorch not yet installed — GPU status unknown (run `uv sync` first).")


def _validate_cloud_key(api_key: str) -> bool:
    """Make a lightweight Ollama Cloud request to verify the key works."""
    try:
        from ollama import Client, ResponseError

        client = Client(
            host=CLOUD_OLLAMA_HOST,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        # A minimal generate call — fast and low cost.
        client.generate(
            model=DEFAULT_CLOUD_MODEL,
            prompt="Say OK.",
            stream=False,
            options={"num_ctx": 32, "temperature": 0},
        )
        return True
    except ResponseError as exc:
        if exc.status_code == 401:
            click.echo("  ✗ API key was rejected (401 Unauthorized).")
        else:
            click.echo(f"  ✗ Ollama Cloud returned an error: {exc}")
        return False
    except Exception as exc:
        click.echo(f"  ✗ Could not reach Ollama Cloud: {exc}")
        return False


def _check_local_ollama(yes: bool = False) -> bool:
    """Check if Ollama is installed and the default model is available."""
    try:
        from ollama import Client, ResponseError

        client = Client()
        models = client.list()
        available = [m.model for m in models.models] if models.models else []
        if any(DEFAULT_LOCAL_MODEL in name for name in available):
            click.echo(f"  ✓ Model {DEFAULT_LOCAL_MODEL} is available locally.")
            return True
        else:
            click.echo(f"  ✗ Model {DEFAULT_LOCAL_MODEL} not found locally.")
            if yes or click.confirm(f"    Pull {DEFAULT_LOCAL_MODEL} now?", default=True):
                click.echo(f"    Pulling {DEFAULT_LOCAL_MODEL}...")
                client.pull(DEFAULT_LOCAL_MODEL)
                click.echo(f"  ✓ {DEFAULT_LOCAL_MODEL} pulled successfully.")
                return True
            return False
    except Exception as exc:
        click.echo(f"  ✗ Could not connect to local Ollama: {exc}")
        click.echo("    Make sure Ollama is installed and running.")
        return False


def run_onboard(
    mode: str | None = None,
    api_key: str | None = None,
    yes: bool = False,
) -> None:
    """Run the onboarding flow."""
    click.echo("")
    click.echo("  Welcome to Anumodana!")
    click.echo("  Let's get your transcription setup configured.")
    click.echo("")

    # Load existing config if any.
    config = load_config()

    # Step 1: mode selection.
    if mode:
        use_cloud = (mode == "cloud")
        click.echo(f"  Mode selected via CLI: {mode}")
    else:
        click.echo("  How would you like to run the AI fixer and review?")
        click.echo("    [1] Cloud (recommended — no GPU needed, free Ollama Cloud tier)")
        click.echo("    [2] Local (requires a GPU and local Ollama)")
        click.echo("")
        mode_choice = click.prompt("  Choice", type=click.IntRange(1, 2), default=1)
        use_cloud = mode_choice == 1
    click.echo("")

    # Step 2: system checks.
    click.echo("  Checking your system...")
    _check_ffmpeg()
    _check_gpu()
    click.echo("")

    if use_cloud:
        # Step 3a: cloud setup.
        if api_key:
            click.echo("  API key provided via CLI.")
            current_api_key = api_key
        else:
            click.echo("  To use Ollama Cloud, you need a free API key.")
            click.echo("    1. Sign up at https://ollama.com")
            click.echo("    2. Get your key at https://ollama.com/settings/keys")
            click.echo("")
            current_api_key = click.prompt("  Paste your API key", hide_input=True).strip()

        click.echo("")
        click.echo("  Validating API key...")
        if _validate_cloud_key(current_api_key):
            click.echo("  ✓ API key validated.")
            config.mode = "cloud"
            config.api_key = current_api_key
            config.fixer.ollama = OllamaConfig(
                model=DEFAULT_CLOUD_MODEL,
                host=CLOUD_OLLAMA_HOST,
            )
            config.review.ollama = OllamaConfig(
                model=DEFAULT_CLOUD_MODEL,
                host=CLOUD_OLLAMA_HOST,
                context_window=32768,
            )
        else:
            click.echo("  API key validation failed. You can re-run `anumodana onboard` later.")
            click.echo("")
            return
    else:
        # Step 3b: local setup.
        click.echo("  Checking local Ollama...")
        if not _check_local_ollama(yes=yes):
            click.echo("  Local setup incomplete. You can re-run `anumodana onboard` later.")
            click.echo("")
            return
        config.mode = "local"
        config.api_key = ""

    # Step 4: save config.
    saved_path = save_config(config)
    click.echo("")
    click.echo(f"  ✓ Config saved to {saved_path}")
    click.echo("")
    click.echo("  You're all set! Run:")
    click.echo('    uv run python -m anumodana pipeline --root "C:\\path\\to\\teachings"')
    click.echo("")
