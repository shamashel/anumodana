---
name: anumodana
description: A skill for running the Anumodana transcription pipeline for Theravada Buddhist talks.
---

# Anumodana Transcription Skill

This skill provides instructions for running the Anumodana transcription pipeline.

## 1. Prerequisites Check

Before running the pipeline, ensure the following are available on the system:

- **uv**: Python dependency manager. Run `uv sync` in the repo root.
- **FFmpeg**: For audio processing. Must be on `PATH` or at `%LOCALAPPDATA%\Programs\ffmpeg`.
- **Ollama**: Must be installed locally (used for model management even in cloud mode).

## 2. Environment Verification

### Check GPU availability (optional — not required)

```powershell
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU — will use CPU (this is fine)')"
```

Parakeet 0.6B runs well on both GPU and CPU. GPU is faster but not required.

### Check Ollama Cloud (preferred for fixer/review)

```powershell
uv run --env-file .env python -c "import os; key = os.environ.get('OLLAMA_API_KEY', ''); print('Cloud API key:', 'configured' if key and key != 'your_api_key_here' else 'NOT SET')"
```

## 3. Choose Execution Mode

Ask the human which mode they prefer. **Recommend cloud unless they have a specific reason to run locally.**

### Cloud mode (default, recommended)

- No GPU needed for the fixer and review passes.
- Requires a free Ollama Cloud API key.
- If the key is not set up yet:
  1. Guide them to copy `.env.example` to `.env`.
  2. Direct them to https://docs.ollama.com/cloud#cloud-api-access to get a free key.
  3. Have them paste the key into `.env`.
- No `--local` flag needed — cloud mode is automatic when the key is set.

### Local mode (requires GPU + local model)

- Requires a GPU with sufficient VRAM for the cleanup model.
- Requires pulling the local model: `ollama pull qwen3.5:4b`
- Run with `--local` flag.

## 4. Choose Directory

Ask the human for the absolute path to the teachings directory they want to process.

- **Check Path**: Verify the path exists and follows the expected structure (at least one collection folder with a `Raw/` subdirectory).

## 5. Data Structure

Ensure the teachings folder follows this structure:

```text
Root/
  Collection Name/
    Raw/               <-- Source audio/video files
    Trimmed/           <-- Processed mono 16kHz MP3s
    Transcript Revision/ <-- Detailed ASR results and human review manifest
```

## 6. Running the Pipeline

The primary entry point is the `anumodana` module:

```powershell
uv run python -m anumodana --root "C:\path\to\your\teachings"
```

For fully local execution:

```powershell
uv run python -m anumodana --root "C:\path\to\your\teachings" --local
```

### Important Flags:
- `--dry-run`: Evaluate what would be processed without writing files.
- `--local`: Run fixer and review models locally instead of via Ollama Cloud.
- `--limit <N>`: Process only the first N files.
- `--skip-fixer`: Skip the AI cleanup pass (faster, but lower quality).
- `--skip-review`: Skip the AI review pass.
- `--chunk-seconds <N>`: Adjust chunk size (default 120). Use 60 for lower memory usage.
- `--keep-models-loaded`: Do not unload models after the run.

## 7. Post-Process Verification

After a run, check the following:
- `_anumodana_review_manifest.csv`: Look for `needs_human_review = True`. (Note: Path columns reference leaf folder names like the date).
- `.review.md`: Read human-readable concerns for specific sessions.
- `.txt`: The final shareable transcript.
