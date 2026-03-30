"""Ollama API client — thin wrapper around the official ``ollama`` library."""

from __future__ import annotations

import json
import logging

from ollama import Client, ResponseError

from anumodana.helpers.config import CLOUD_OLLAMA_HOST, OllamaConfig

logger = logging.getLogger("anumodana")


def build_client(config: OllamaConfig, api_key: str = "") -> Client:
    """Build an :class:`ollama.Client` for local or cloud mode."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return Client(host=config.host, headers=headers)


def call_ollama(
    client: Client,
    config: OllamaConfig,
    prompt: str,
    format_schema: dict[str, object],
) -> dict[str, object]:
    """Send a ``generate`` request and return the parsed JSON response."""
    try:
        response = client.generate(
            model=config.model,
            prompt=prompt,
            format=format_schema,
            options={"temperature": config.temperature, "num_ctx": config.context_window},
            stream=False,
            think=False,
        )
    except ResponseError as exc:
        if exc.status_code == 429 and CLOUD_OLLAMA_HOST in config.host:
            raise RuntimeError(
                "Ollama Cloud rate limit exceeded. "
                "See https://ollama.com/pricing for details. "
                f"Original error: {exc}"
            ) from exc
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to reach Ollama at {config.host}: {exc}") from exc

    raw_response = response.response
    if not raw_response or not raw_response.strip():
        raise RuntimeError("Ollama returned an empty response.")

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama did not return valid JSON: {raw_response[:400]}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("Ollama returned JSON, but not an object.")
    return parsed


def unload_model(client: Client, config: OllamaConfig) -> None:
    """Ask Ollama to evict a model from memory (local only)."""
    if CLOUD_OLLAMA_HOST in config.host:
        return
    try:
        client.generate(model=config.model, prompt="", keep_alive=0)
    except Exception:
        pass
