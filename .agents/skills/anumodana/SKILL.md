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

## 2. First-Time Setup

Run the onboarding wizard:

```powershell
uv run python -m anumodana onboard
```

This will:
- Ask the user to choose cloud or local mode
- Check for FFmpeg and GPU availability
- Walk through API key setup (cloud) or model pulling (local)
- Save config to `~/.config/anumodana/config.json`

**Recommend cloud mode unless the user has a specific reason to run locally.**

### If already configured

Check if config exists:

```powershell
uv run python -c "from anumodana.config import load_config; c = load_config(); print(f'Mode: {c.mode}'); print(f'API key: {\"configured\" if c.api_key else \"NOT SET\"}')"
```

## 3. Choose Directory

Ask the human for the absolute path to the teachings directory they want to process.

- **Check Path**: Verify the path exists and follows the expected structure (at least one collection folder with a `Raw/` subdirectory).

## 4. Data Structure

Ensure the teachings folder follows this structure:

```text
Root/
  Collection Name/
    Raw/               <-- Source audio/video files
    Trimmed/           <-- Processed mono 16kHz MP3s
    Transcript Revision/ <-- Detailed ASR results and human review manifest
```

## 5. Running the Pipeline

### Full pipeline (transcribe → fix → review):

```powershell
uv run python -m anumodana pipeline --root "C:\path\to\your\teachings"
```

### Transcription only (no fixer, no review):

```powershell
uv run python -m anumodana transcribe --root "C:\path\to\your\teachings"
```

### For fully local execution:

```powershell
uv run python -m anumodana pipeline --root "C:\path\to\your\teachings" --local
```

### Important Flags:
- `--dry-run`: Evaluate what would be processed without writing files.
- `--local`: Run fixer and review models locally instead of via Ollama Cloud.
- `--limit <N>`: Process only the first N files.
- `--skip-fixer`: Skip the AI fixer pass (faster, but lower quality).
- `--skip-review`: Skip the AI review pass.
- `--keep-models-loaded`: Do not unload models after the run.
- `--verbose`: Show verbose library diagnostics.

### Standalone commands:

```powershell
uv run python -m anumodana fix "C:\path\to\input.vtt"
uv run python -m anumodana review "C:\path\to\raw.vtt" "C:\path\to\cleaned.vtt"
```

## 6. Post-Process Verification

After a run, check the following:
- `_anumodana_review_manifest.csv`: Look for `needs_human_review = True`. (Note: Path columns reference leaf folder names like the date).
- `.review.md`: Read human-readable concerns for specific sessions.
- `.txt`: The final shareable transcript.
