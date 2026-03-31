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

## 2. First-Time Setup (Agent-Centric)

As an AI agent, you should handle onboarding non-interactively using CLI flags.

### 2.1 Check if already configured

Before running the full setup, check if a valid configuration already exists:

```powershell
uv run python -c "from anumodana.helpers.config import load_config; c = load_config(); print(f'Mode: {c.mode}'); print(f'API key: {\"configured\" if c.api_key else \"NOT SET\"}')"
```

### 2.2 Automated Onboarding

If the configuration is missing or invalid (e.g., cloud mode selected but no API key), run the onboarding command with non-interactive flags. 

**Standard Cloud Setup (Recommended)**:
Ask the human for their Ollama Cloud API key, then run:

```powershell
uv run python -m anumodana onboard --mode cloud --api-key "YOUR_API_KEY" --yes
```

**Standard Local Setup**:
If the user specifically requests local execution and has a GPU, run:

```powershell
uv run python -m anumodana onboard --mode local --yes
```

> [!IMPORTANT]
> Always include the `--yes` flag when running onboarding as an agent. This avoids interactive confirmation prompts for operations like pulling local models.

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

### Transcription pipeline (transcribe → fix):

```powershell
uv run python -m anumodana pipeline --root "C:\path\to\your\teachings"
```

### Full pipeline (transcribe → fix → review):

```powershell
uv run python -m anumodana pipeline --review --root "C:\path\to\your\teachings"
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
- `--review`: Enable the AI review pass and generate a manifest.
- `--local`: Run fixer and review models locally instead of via Ollama Cloud.
- `--limit <N>`: Process only the first N files.
- `--keep-models-loaded`: Do not unload models after the run.
- `--verbose`: Show verbose library diagnostics.

### Standalone commands:

```powershell
uv run python -m anumodana fix "C:\path\to\input.vtt"
uv run python -m anumodana review "C:\path\to\raw.vtt" "C:\path\to\cleaned.vtt"
```

## 6. Post-Process Verification

After a run, check the following:
- `_anumodana_review_manifest.csv`: Look for `needs_human_review = True`. (Only if `--review` was used).
- `.review.md`: Read human-readable concerns for specific sessions. (Only if `--review` was used).
- `.txt`: The final shareable transcript.
