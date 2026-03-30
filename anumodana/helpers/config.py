"""Centralized configuration for Anumodana.

Config lives at ``~/.config/anumodana/config.json``.  It is created by
``anumodana onboard`` and read on every pipeline run.  CLI flags override
values in the config file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger("anumodana")

CONFIG_DIR = Path.home() / ".config" / "anumodana"
CONFIG_PATH = CONFIG_DIR / "config.json"

# ---------------------------------------------------------------------------
# Default model names
# ---------------------------------------------------------------------------
DEFAULT_LOCAL_MODEL = "qwen3.5:4b"
DEFAULT_CLOUD_MODEL = "qwen3.5"
DEFAULT_TRANSCRIPTION_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

LOCAL_OLLAMA_HOST = "http://127.0.0.1:11434"
CLOUD_OLLAMA_HOST = "https://ollama.com"


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------
class OllamaConfig(BaseModel):
    """Shared Ollama connection & generation settings."""

    model: str = DEFAULT_LOCAL_MODEL
    host: str = LOCAL_OLLAMA_HOST
    temperature: float = 0.1
    context_window: int = 8192


class FixerConfig(BaseModel):
    """Fixer-pass overrides on top of shared Ollama settings."""

    ollama: OllamaConfig = OllamaConfig()
    batch_size: int = 16
    max_batch_characters: int = 3200
    max_prompt_tokens: int = 2600


class ReviewConfig(BaseModel):
    """Review-pass overrides on top of shared Ollama settings."""

    ollama: OllamaConfig = OllamaConfig(context_window=32768)


class TranscriptionConfig(BaseModel):
    """Parakeet ASR settings."""

    model: str = DEFAULT_TRANSCRIPTION_MODEL
    chunk_seconds: int = 120


class AnumodanaConfig(BaseModel):
    """Root config — serialised to / from ``config.json``."""

    mode: Literal["cloud", "local"] = "cloud"
    api_key: str = ""
    transcription: TranscriptionConfig = TranscriptionConfig()
    fixer: FixerConfig = FixerConfig()
    review: ReviewConfig = ReviewConfig()


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------
def load_config() -> AnumodanaConfig:
    """Read config from disk, returning defaults if the file is missing."""
    if not CONFIG_PATH.exists():
        return AnumodanaConfig()
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return AnumodanaConfig.model_validate(data)
    except Exception as exc:
        logger.warning("Failed to load config from %s: %s — using defaults", CONFIG_PATH, exc)
        return AnumodanaConfig()


def save_config(config: AnumodanaConfig) -> Path:
    """Write config to disk, creating the directory if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        config.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return CONFIG_PATH


# ---------------------------------------------------------------------------
# Runtime resolution
# ---------------------------------------------------------------------------
CLOUD_BATCH_MULTIPLIER = 5


def resolve_runtime_config(
    config: AnumodanaConfig,
    *,
    local: bool = False,
) -> AnumodanaConfig:
    """Build a fully-resolved config, applying cloud-mode adjustments.

    The returned config is a *copy* — the original is not mutated.
    """
    data = config.model_dump()

    if local:
        data["mode"] = "local"

    is_cloud = data["mode"] == "cloud" and data["api_key"]

    if is_cloud:
        # Point both fixer and review at the cloud host + cloud model.
        for section in ("fixer", "review"):
            data[section]["ollama"]["host"] = CLOUD_OLLAMA_HOST
            if data[section]["ollama"]["model"] == DEFAULT_LOCAL_MODEL:
                data[section]["ollama"]["model"] = DEFAULT_CLOUD_MODEL

        # Cloud context windows are large — disable batching and scale limits.
        fixer = data["fixer"]
        if fixer["batch_size"] == FixerConfig().batch_size:
            fixer["batch_size"] = 0
        if fixer["max_batch_characters"] == FixerConfig().max_batch_characters:
            fixer["max_batch_characters"] *= CLOUD_BATCH_MULTIPLIER
        if fixer["max_prompt_tokens"] == FixerConfig().max_prompt_tokens:
            fixer["max_prompt_tokens"] *= CLOUD_BATCH_MULTIPLIER
    elif data["mode"] == "cloud" and not data["api_key"]:
        # Wanted cloud but no key — fall back to local silently.
        data["mode"] = "local"

    return AnumodanaConfig.model_validate(data)
