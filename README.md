# Anumodana

Anumodana is a transcription pipeline for Theravada Buddhist talks.

It takes a folder of audio or video recordings and produces clean, shareable transcripts — with Buddhist terminology, Pali chants, and lineage-specific names handled automatically.

Right now, it is primarily aimed at English-speaking communities. It works best when the main body of the talk is in English, even if it includes some Pali or Buddhist terms.

For each recording, the pipeline writes:

- a same-name mono 16 kHz `.mp3`
- a human-readable transcript: `.txt`
- revision artifacts under `Transcript Revision/`: `.parakeet.raw.vtt`, cleaned `.vtt`, `.review.json`, `.review.md`
- a revision summary file: `_anumodana_review_manifest.csv`

The pipeline:

1. media file → `.mp3`
2. `nvidia/parakeet-tdt-0.6b-v3` (local transcription — uses your GPU if you have one, otherwise runs on CPU)
3. AI fixer pass (Ollama Cloud by default, or local with `--local`)
4. AI review pass (Ollama Cloud by default, or local with `--local`)

## Who This Is For

This project is meant to be usable by anyone who wants to help transcribe Dhamma talks:

- monastics or laypeople with little or no technical background, working through an AI assistant
- people who are comfortable running commands in a terminal
- technically inclined users (monks or laypeople from IT, programming, or similar backgrounds)

**No dedicated GPU is required.** The transcription model is small enough to run on a normal computer's CPU, and the AI cleanup and review steps run in the cloud by default.

If you are not technical, the easiest path is usually:

1. Install the prerequisites below once (or ask someone to help you).
2. Open this project in your AI assistant (Codex, Claude Code/Coworker, or similar).
3. Ask it to verify setup, then run `python -m anumodana` on your teachings folder.
4. Ask it to summarize anything flagged for human review in `_anumodana_review_manifest.csv`.

## Current Limitations

- This workflow currently works best for English talks.
- `nvidia/parakeet-tdt-0.6b-v3` has been a strong option for English transcription, but it is not a good fit for Thai-heavy content as currently configured.
- Pali chants and lineage-specific terminology can still need cleanup or human review, especially at the start of talks.
- If a talk contains long Thai sections, a different ASR model or a future fine-tuned model may be a better choice.

## Quick Start

### 1. Install prerequisites

You need:

- [uv](https://docs.astral.sh/uv/) (Python project manager)
- Python 3.12 (uv can install this for you)
- [FFmpeg](https://ffmpeg.org/) (audio processing)
- [Ollama](https://ollama.com/) (installed locally — needed even for cloud mode, as the local `ollama` command is used for model management)

Once you have `uv`, run this in the project folder to install Python and all dependencies:

```powershell
uv sync
```

FFmpeg should either be on your `PATH`, or installed under:

```text
%LOCALAPPDATA%\Programs\ffmpeg
```

### 2. Choose how to run the AI cleanup and review

#### Option A: Cloud (recommended — easiest, no GPU needed)

This is the default. The transcription step runs locally on your machine (no internet needed for that part), but the AI cleanup and review steps use Ollama Cloud's free tier.

1. Sign up at [ollama.com](https://ollama.com).
2. Get an API key at [Ollama Cloud API](https://docs.ollama.com/cloud#cloud-api-access).
3. Copy the `.env.example` file to `.env` in the project root.
4. Paste your key in `.env`:
   ```
   OLLAMA_API_KEY="your_api_key_here"
   ```

That's it. The pipeline will automatically use cloud models when it sees this key.

#### Option B: Fully local (requires a GPU with enough VRAM for the cleanup model)

If you prefer to run everything on your own machine:

1. Pull the local cleanup model:
   ```powershell
   ollama pull qwen3.5:4b
   ```
2. Run the pipeline with the `--local` flag:
   ```powershell
   uv run python -m anumodana --local
   ```

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

### 4. Run the pipeline

```powershell
uv run python -m anumodana
```

To run a different parent folder:

```powershell
uv run python -m anumodana --root "C:\path\to\teachings"
```

You can also point directly at a single collection folder or a `Trimmed` folder to process just one collection.

Keep models loaded after the run:

```powershell
uv run python -m anumodana --keep-models-loaded
```

Show verbose library diagnostics during transcription:

```powershell
uv run python -m anumodana --verbose
```

### Choosing chunk size (advanced)

The Parakeet transcription step uses `--chunk-seconds` to decide how much audio to send through the model at once.

The default is `120`, and that is the recommended starting point.

Practical rule of thumb:

- `60` seconds is a safer choice if you want to minimize RAM or VRAM usage.
- `120` seconds is the recommended default.
- `180`+ seconds can be reasonable on high-VRAM GPUs, but test carefully.

<details>
<summary>GPU VRAM observations (RTX 4090)</summary>

- `30` seconds: about `+200 MiB` during transcription
- `60` seconds: about `+500 MiB`
- `120` seconds: about `+1.0 GiB`
- `180` seconds: about `+1.7 GiB`
- `240` seconds: about `+2.9 GiB`

These numbers are a rough heuristic. PyTorch and CUDA caching can make the baseline move around between runs, and results differ based on driver mode, background GPU usage, and whether the GPU is also running the desktop.

On a display-attached GPU, high VRAM pressure can make the whole computer feel sluggish even if the run does not crash. If you notice lag, go smaller.
</details>

If you are unsure, stay with the default:

```powershell
uv run python -m anumodana
```

If you want a more conservative setting:

```powershell
uv run python -m anumodana --chunk-seconds 60
```

## What The Files Mean

For each teaching, the pipeline writes:

- `session.mp3`
  A same-name mono 16 kHz MP3 audio copy for listening, sharing, or archiving.
- `session.txt`
  A human-readable transcript for sharing alongside the media.

Under each collection's sibling `Transcript Revision/...` tree, the pipeline also writes:

- `session.parakeet.raw.vtt`
  The direct ASR output from Parakeet before cleanup.
- `session.vtt`
  The cleaned timing-aligned subtitle file used for review.
- `session.review.json`
  Structured review data.
- `session.review.md`
  A human-readable review note.

At the root of that revision tree, it also writes:

- `_anumodana_review_manifest.csv`
  A one-row-per-session summary. Path columns (source_path, etc.) reference the leaf folder name (e.g. the date) for easier readability.

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

If you already have both a raw transcript and a cleaned timing-aligned transcript:

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

2. Check if a GPU is available (optional — not required):

```powershell
uv run python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU — will use CPU (this is fine)')"
```

3. Check if Ollama Cloud is configured (preferred):

```powershell
uv run python -c "import os; key = os.environ.get('OLLAMA_API_KEY', ''); print('Cloud API key:', 'configured' if key and key != 'your_api_key_here' else 'NOT SET — see .env.example')"
```

If the key is not set, guide the user to:
- Copy `.env.example` to `.env`
- Get a free API key at https://docs.ollama.com/cloud#cloud-api-access
- Paste it into `.env`

4. If the user wants to run fully local instead (`--local`), verify Ollama is installed and the cleanup model exists:

```powershell
ollama list
```

Look for `qwen3.5:4b`. If missing:

```powershell
ollama pull qwen3.5:4b
```

5. Verify FFmpeg is available:

```powershell
ffmpeg -version
ffprobe -version
```

### Basic verification

Check the pipeline without modifying data:

```powershell
uv run python -m anumodana --root "<parent folder or collection folder>" --dry-run
```

If the user wants a real run:

```powershell
uv run python -m anumodana --root "<parent folder or collection folder>"
```

For fully local runs:

```powershell
uv run python -m anumodana --root "<parent folder or collection folder>" --local
```

### What to tell the human

Surface these things clearly:

- whether CUDA is being used for transcription, or if it fell back to CPU (both work — GPU is just faster)
- whether the pipeline is using Ollama Cloud or local Ollama for fixer/review
- whether the required models were downloaded successfully
- which file is currently being processed
- where `_anumodana_review_manifest.csv` was written
- whether any sessions were flagged with `needs_human_review = true`
- any failures involving missing FFmpeg, missing Ollama, missing model files, or missing API keys

If the run succeeds, direct the human to:

- the `.txt` files for normal reading and sharing
- the `.review.md` files for human-readable concerns
- `_anumodana_review_manifest.csv`: Look for `needs_human_review = True`. (Note: Path columns reference leaf folder names like the date).
