from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def call_ollama(
    url: str,
    model: str,
    prompt: str,
    temperature: float,
    context_window: int,
    format_schema: dict[str, object] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_ctx": context_window,
        },
    }
    if format_schema is not None:
        body["format"] = format_schema

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
