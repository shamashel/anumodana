---
name: anumodana
description: A skill for running the Anumodana transcription pipeline for Theravada Buddhist talks.
---

# Anumodana Transcription Skill

This skill provides instructions for running the Anumodana transcription pipeline.

## 1. Prerequisites Check

Before running the pipeline, ensure the following are available on the system:

- **uv**: Python dependency manager.
- **FFmpeg**: For audio processing.
- **Ollama**: For cleanup and review passes.
- **Model**: `qwen3.5:4b` must be pulled in Ollama (`ollama pull qwen3.5:4b`).
- **CUDA**: Strongly recommended for performance.

## 2. Environment Verification

Run the following command to verify the environment and CUDA status:

```powershell
uv run python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 3. Choose Directory

Before proceeding with the run, ask the human for the absolute path to the teachings directory they want to process.

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

## 4. Running the Pipeline

The primary entry point is the `anumodana` module:

```powershell
uv run python -m anumodana --root "C:\path\to\your\teachings"
```

### Important Flags:
- `--dry-run`: Evaluate what would be processed without writing files.
- `--limit <N>`: Process only the first N files.
- `--skip-qwen`: Skip the AI cleanup pass (faster, but lower quality).
- `--skip-review`: Skip the AI review pass.
- `--chunk-seconds <N>`: Adjust VRAM usage (default 120, use 60 for lower VRAM).

## 5. Post-Process Verification

After a run, check the following:
- `_anumodana_review_manifest.csv`: Look for `needs_human_review = True`.
- `.review.md`: Read human-readable concerns for specific sessions.
- `.txt`: The final shareable transcript.
