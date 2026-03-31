# Anumodana

Anumodana is a transcription pipeline for Theravada Buddhist talks.

It takes a folder of audio or video recordings and produces clean, shareable transcripts — with Buddhist terminology, Pali chants, and lineage-specific names handled automatically.

Right now, it is primarily aimed at English-speaking communities. It works best when the main body of the talk is in English, even if it includes some Pali or Buddhist terms.

For each recording, the pipeline writes:

- a same-name mono 16 kHz `.mp3`
- a human-readable transcript: `.txt`
- revision artifacts under `Transcript Revision/` (if `--review` is used): `.parakeet.raw.vtt`, cleaned `.vtt`, `.review.json`, `.review.md`
- a revision summary file (if `--review` is used): `_anumodana_review_manifest.csv`

1. media file → `.mp3`
2. `nvidia/parakeet-tdt-0.6b-v3` (local transcription — uses your GPU if you have one, otherwise runs on CPU)
3. AI fixer pass (Ollama Cloud by default, or local with `--local`)
4. AI review pass (opt-in with `--review`)

## Who This Is For

This project is meant to be usable by anyone who wants to help transcribe Dhamma talks:

- monastics or laypeople with little or no technical background, working through an AI assistant
- people who are comfortable running commands in a terminal
- technically inclined users (monks or laypeople from IT, programming, or similar backgrounds)

**No dedicated GPU is required.** The transcription model is small enough to run on a normal computer's CPU, and the AI fixer and review steps run in the cloud by default.

If you are not technical, the easiest path is usually:

1. Install the prerequisites below once (or ask someone to help you).
2. Open this project in your AI assistant (Codex, Claude Code/Coworker, or similar).
3. Ask it to run `anumodana onboard` and follow the prompts.
4. Ask it to run `anumodana pipeline --root "C:\path\to\teachings" --review`.
5. Ask it to summarize anything flagged for human review in `_anumodana_review_manifest.csv`.

## Current Limitations

- This workflow currently works best for English talks.
- `nvidia/parakeet-tdt-0.6b-v3` has been a strong option for English transcription, but it is not a good fit for Thai-heavy content as currently configured.
- Pali chants and lineage-specific terminology can still need fixer or human review, especially at the start of talks.
- If a talk contains long Thai sections, a different ASR model or a future fine-tuned model may be a better choice.

## Quick Start

### 1. Install prerequisites

You need:

- [uv](https://docs.astral.sh/uv/) (Python project manager)
- Python 3.12 (uv can install this for you)
- [FFmpeg](https://ffmpeg.org/) (audio processing)

Once you have `uv`, run this in the project folder to install Python and all dependencies:

```powershell
uv sync
```

FFmpeg should either be on your `PATH`, or installed under:

```text
%LOCALAPPDATA%\Programs\ffmpeg
```

### 2. Run first-time setup

```powershell
uv run python -m anumodana onboard
```

The onboard wizard will:
- Ask whether you want to use cloud or local mode
- Check for FFmpeg and GPU availability
- Walk you through getting an API key (cloud) or pulling the local model
- Save your configuration to `~/.config/anumodana/config.json`

### 3. Organize your teachings

By default, the script looks in:

```text
~/Downloads/Ajahn Wade Recordings
```

On Windows, that is usually something like:

```text
C:\Users\<you>\Downloads\Ajahn Wade Recordings
```

The batch pipeline expects a collection layout like this:

```text
Root/
  Collection Name/
    Raw/
    Trimmed/
    Transcript Revision/
```

The collection name can be anything. The important part is the `Raw/`, `Trimmed/`, and `Transcript Revision/` subdirectory structure.

### `anumodana pipeline` (default)

Run the transcription pipeline: transcribe → fix. Use `--review` to also run the review pass.

```powershell
uv run python -m anumodana pipeline --root "C:\path\to\teachings"
uv run python -m anumodana pipeline --root "C:\path\to\teachings" --review
uv run python -m anumodana pipeline --root "C:\path\to\teachings" --local
uv run python -m anumodana pipeline --root "C:\path\to\teachings" --dry-run
uv run python -m anumodana pipeline --root "C:\path\to\teachings" --limit 2
uv run python -m anumodana pipeline --root "C:\path\to\teachings" --keep-models-loaded
uv run python -m anumodana pipeline --root "C:\path\to\teachings" --verbose
```

### `anumodana transcribe`

Run only the ASR step (Parakeet → raw VTT → plain text). No fixer, no review.

```powershell
uv run python -m anumodana transcribe --root "C:\path\to\teachings"
```


### `anumodana fix`

Run the AI fixer on an existing raw VTT file.

```powershell
uv run python -m anumodana fix "C:\path\to\input.vtt"
uv run python -m anumodana fix "C:\path\to\input.vtt" --local
uv run python -m anumodana fix "C:\path\to\input.vtt" --glossary-file "C:\path\to\terms.txt"
```

### `anumodana review`

Run the AI review comparing a raw transcript to a cleaned one.

```powershell
uv run python -m anumodana review "C:\path\to\session.parakeet.raw.vtt" "C:\path\to\session.vtt"
```

### `anumodana onboard`

Interactive setup wizard — choose cloud or local, configure API keys, verify prerequisites.

```powershell
uv run python -m anumodana onboard
```

## Configuration

All configuration lives in `~/.config/anumodana/config.json`. This file is created by `anumodana onboard` and can be edited by hand for advanced tuning.

The config includes:
- `mode`: `"cloud"` or `"local"`
- `api_key`: Ollama Cloud API key (for cloud mode)
- `transcription`: ASR model and chunk size
- `fixer`: model, batch size, token limits
- `review`: model, context window

## What The Files Mean

For each teaching, the pipeline writes:

- `session.mp3`
  A same-name mono 16 kHz MP3 audio copy for listening, sharing, or archiving.
- `session.txt`
  A human-readable transcript for sharing alongside the media.

Under each collection's sibling `Transcript Revision/...` tree, the pipeline also writes:

- `session.parakeet.raw.vtt`
  The direct ASR output from Parakeet before the fixer pass.
- `session.vtt`
  The cleaned timing-aligned subtitle file used for review.
- `session.review.json`
  Structured review data (generated if `--review` is used).
- `session.review.md`
  A human-readable review note (generated if `--review` is used).

At the root of that revision tree, it also writes:

- `_anumodana_review_manifest.csv`
  A one-row-per-session summary (generated if `--review` is used). Path columns reference the leaf folder name (e.g. the date) for easier readability.

## Glossaries

The default fixer stack loads these files in order:

- `glossaries/core_chants.txt`
- `glossaries/core_theravada_terms.txt`
- `glossaries/lineages/ajahn_chah.txt`
- `glossaries/local_teachers_and_places.txt`

This makes it easier to adapt the project for a different monastery, lineage, or local community.

## For AI Agents

If you are an AI agent, please see the SKILL.md located at `.agents/skills/anumodana/SKILL.md` for a comprehensive guide on running this pipeline, handling onboarding, and troubleshooting.

### Setup checklist

1. Run `uv sync` in the repo root.

2. Run `anumodana onboard` to configure the pipeline:

```powershell
uv run python -m anumodana onboard
```

3. Verify FFmpeg is available:

```powershell
ffmpeg -version
ffprobe -version
```

### Basic verification

Check the pipeline without modifying data:

```powershell
uv run python -m anumodana pipeline --root "<parent folder or collection folder>" --dry-run
```

If the user wants a real run:

```powershell
uv run python -m anumodana pipeline --root "<parent folder or collection folder>"
```

If the user specifically wants a review pass:

```powershell
uv run python -m anumodana pipeline --review --root "<parent folder or collection folder>"
```

### What to tell the human

Surface these things clearly:

- whether CUDA is being used for transcription, or if it fell back to CPU (both work — GPU is just faster)
- whether the pipeline is using Ollama Cloud or local Ollama for fixer/review
- whether the required models were downloaded successfully
- which file is currently being processed
- where `_anumodana_review_manifest.csv` was written (if using `--review`)
- whether any sessions were flagged with `needs_human_review = true` (if using `--review`)
- any failures involving missing FFmpeg, missing Ollama, missing model files, or missing API keys

If the run succeeds, direct the human to:

- the `.txt` files for normal reading and sharing
- the `.review.md` files for human-readable concerns
- `_anumodana_review_manifest.csv`: Look for `needs_human_review = True`. (Note: Path columns reference leaf folder names like the date).
