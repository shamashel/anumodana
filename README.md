# Anumodana

Anumodana is a local transcription pipeline for Theravada Buddhist talks.

Right now, it is primarily aimed at English-speaking communities. It works best when the main body of the talk is in English, even if it includes some Pali or Buddhist terminology.

It takes a folder of video or audio files and produces:

- a same-name mono 16 kHz `.mp3`
- a raw Parakeet transcript: `.parakeet.raw.vtt`
- a cleaned subtitle file: `.vtt`
- a review file: `.review.json`
- a human-readable review note: `.review.md`
- a root-level summary file: `_anumodana_review_manifest.csv`

The current pipeline is:

1. media file
2. `.mp3`
3. `nvidia/parakeet-tdt-0.6b-v3`
4. local `qwen3.5:4b` cleanup pass
5. local review pass

## Current Limitations

- This workflow currently works best for English talks.
- `nvidia/parakeet-tdt-0.6b-v3` has been a strong local option for English transcription here, but it is not a good fit for Thai-heavy content in this project as currently configured.
- Pali chants and lineage-specific terminology can still need cleanup or human review, especially at the start of talks.
- If a talk contains long Thai sections, a different ASR model or a future fine-tuned model may be a better choice.

## Who This Is For

This project is meant to be usable by:

- monastics or laypeople who want help transcribing teachings locally
- technically inclined users who are comfortable running commands
- people working with an AI coding assistant such as Codex, Claude Code/Coworker, or similar tools

If you are not technical, the easiest path is usually:

1. Install the prerequisites below once.
2. Open this project in your AI assistant.
3. Ask it to verify setup, then run `python -m anumodana` on your teachings folder.
4. Ask it to summarize anything flagged for human review in `_anumodana_review_manifest.csv`.

## Quick Start

### 1. Install prerequisites

You need:

- `uv`
- Python 3.12
- FFmpeg
- [Ollama](https://ollama.com/)
- the local model `qwen3.5:4b`

If you already have `uv`, the repo can set up Python and dependencies with:

```powershell
uv sync
```

Then pull the local cleanup model:

```powershell
ollama pull qwen3.5:4b
```

FFmpeg should either be on your `PATH`, or installed under:

```text
%LOCALAPPDATA%\Programs\ffmpeg
```

### 2. Put your teachings in one folder tree

By default, the script looks in:

```text
~/Downloads/Trimmed
```

On Windows, that is usually something like:

```text
C:\Users\<you>\Downloads\Trimmed
```

### 3. Run the pipeline

```powershell
uv run python -m anumodana
```

To run a different folder:

```powershell
uv run python -m anumodana --root "C:\path\to\teachings"
```

The repo now has one CLI entry point:

```powershell
uv run python -m anumodana --help
```

The full pipeline is the default behavior. Standalone cleanup and review live under:

```powershell
uv run python -m anumodana cleanup --help
uv run python -m anumodana review --help
```

## If You Are Using An AI Helper

You can usually tell your AI assistant something like:

```text
Please verify this repo is set up correctly, make sure CUDA is available, then run the transcription pipeline on my teachings folder and summarize anything that needs human review.
```

That is often the smoothest path for non-technical users.

## Common Commands

Dry run without writing files:

```powershell
uv run python -m anumodana --dry-run
```

Only process the first 5 files:

```powershell
uv run python -m anumodana --limit 5
```

Overwrite existing outputs:

```powershell
uv run python -m anumodana --overwrite
```

Skip cleanup:

```powershell
uv run python -m anumodana --skip-qwen
```

Skip review:

```powershell
uv run python -m anumodana --skip-review
```

Keep models loaded after the run:

```powershell
uv run python -m anumodana --keep-models-loaded
```

Show verbose library diagnostics during transcription:

```powershell
uv run python -m anumodana --verbose
```

## What The Files Mean

For each teaching, the pipeline writes:

- `session.mp3`
  A same-name mono 16 kHz MP3 audio copy for listening, sharing, or archiving.
- `session.parakeet.raw.vtt`
  The direct ASR output from Parakeet before cleanup.
- `session.vtt`
  The cleaned subtitle file after the local Qwen pass.
- `session.review.json`
  Structured review data.
- `session.review.md`
  A human-readable review note.

At the root of the run, it also writes:

- `_anumodana_review_manifest.csv`
  A one-row-per-session summary of outputs and review status.

## Standalone Cleanup

If you already have a raw `.vtt`, you can run just the cleanup step:

```powershell
uv run python -m anumodana cleanup "C:\path\to\input.vtt"
```

Useful options:

```powershell
uv run python -m anumodana cleanup "C:\path\to\input.vtt" --batch-size 0
uv run python -m anumodana cleanup "C:\path\to\input.vtt" --glossary-file "C:\path\to\my_lineage_terms.txt"
uv run python -m anumodana cleanup "C:\path\to\input.vtt" --no-default-glossaries
uv run python -m anumodana cleanup "C:\path\to\input.vtt" --keep-model-loaded
```

## Standalone Review

If you already have both a raw transcript and a cleaned transcript:

```powershell
uv run python -m anumodana review "C:\path\to\session.parakeet.raw.vtt" "C:\path\to\session.vtt"
```

The review output includes:

- `summary`
- `review_notes`
- `concerns`
- `needs_human_review`

## Glossaries

The default correction stack loads these files in order:

- `glossaries/core_chants.txt`
- `glossaries/core_theravada_terms.txt`
- `glossaries/lineages/ajahn_chah.txt`
- `glossaries/local_teachers_and_places.txt`

This makes it easier to adapt the project for a different monastery, lineage, or local community.

## For AI Agents

Use this section as the fast path.

### Setup checklist

1. Run `uv sync` in the repo root.
2. Verify CUDA is available:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

3. Verify Ollama is installed and the cleanup model exists:

```powershell
ollama list
```

Look for `qwen3.5:4b`. If missing:

```powershell
ollama pull qwen3.5:4b
```

4. Verify FFmpeg is available:

```powershell
ffmpeg -version
ffprobe -version
```

### Basic verification

Check the pipeline without modifying data:

```powershell
uv run python -m anumodana --root "<teachings folder>" --dry-run
```

If the user wants a real run:

```powershell
uv run python -m anumodana --root "<teachings folder>"
```

### What to tell the human

Surface these things clearly:

- whether CUDA is actually being used or if the run fell back to CPU
- whether the required models were downloaded successfully
- which file is currently being processed
- where `_anumodana_review_manifest.csv` was written
- whether any sessions were flagged with `needs_human_review = true`
- any failures involving missing FFmpeg, missing Ollama, missing model files, or CUDA not being available

If the run succeeds, direct the human to:

- the cleaned `.vtt` files for normal use
- the `.review.md` files for human-readable concerns
- `_anumodana_review_manifest.csv` for the overall summary
