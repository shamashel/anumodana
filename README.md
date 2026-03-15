# Anumodana

Local transcript tools for Theravada Buddhist talks.

The current pipeline is:

- video or audio file
- same-name mono 16 kHz `.wav`
- `nvidia/parakeet-tdt-0.6b-v3`
- local `qwen3.5:9b` cleanup pass with Buddhist and lineage glossaries
- same-name cleaned `.vtt`
- same-name raw `.parakeet.raw.vtt`
- same-name `.review.json`
- same-name `.review.md`
- root-level `_anumodana_review_manifest.csv`

## What Is Here

- `batch_parakeet_tree.py`
  Walks a directory tree, picks one source per same-name media set, creates `.wav` files, runs Parakeet, writes a raw transcript, applies the Qwen cleanup pass, writes review artifacts, and updates a manifest CSV.
- `llm_correct_vtt.py`
  Runs the local Ollama correction step against a `.vtt` file.
- `review_vtt.py`
  Reviews a cleaned `.vtt` against its raw ASR transcript and emits structured review data plus a human-readable report.
- `glossaries/`
  Modular glossary files for chants, Theravada terminology, Ajahn Chah lineage terms, and local teachers or places.

## Install

```powershell
uv python install 3.12 --default
uv venv .venv --python 3.12
uv pip install -r requirements.txt
```

You also need:

- an FFmpeg build on `PATH`, or under `%LOCALAPPDATA%\Programs\ffmpeg`
- [Ollama](https://ollama.com/) with `qwen3.5:9b` pulled locally

Example:

```powershell
ollama pull qwen3.5:9b
```

## Batch Run

Default root:

```text
C:\Users\Shamash\Downloads\Trimmed
```

Run the full pipeline:

```powershell
uv run python batch_parakeet_tree.py
```

Useful flags:

```powershell
uv run python batch_parakeet_tree.py --dry-run
uv run python batch_parakeet_tree.py --limit 5
uv run python batch_parakeet_tree.py --overwrite
uv run python batch_parakeet_tree.py --skip-qwen
uv run python batch_parakeet_tree.py --skip-review
uv run python batch_parakeet_tree.py --keep-models-loaded
```

Per source file, the batch script now writes:

- `session.wav`
- `session.parakeet.raw.vtt`
- `session.vtt`
- `session.review.json`
- `session.review.md`

At the root of the run, it also writes:

- `_anumodana_review_manifest.csv`

By default, the batch script unloads both the Parakeet model and the Ollama model when it exits.

## Standalone Cleanup

If you already have a raw `.vtt`, run just the cleanup pass:

```powershell
uv run python llm_correct_vtt.py "C:\path\to\input.vtt"
```

Useful flags:

```powershell
uv run python llm_correct_vtt.py "C:\path\to\input.vtt" --batch-size 0
uv run python llm_correct_vtt.py "C:\path\to\input.vtt" --glossary-file "C:\path\to\my_lineage_terms.txt"
uv run python llm_correct_vtt.py "C:\path\to\input.vtt" --no-default-glossaries
uv run python llm_correct_vtt.py "C:\path\to\input.vtt" --keep-model-loaded
```

## Standalone Review

If you already have both a raw ASR transcript and a cleaned transcript, run the review pass like this:

```powershell
uv run python review_vtt.py "C:\path\to\session.parakeet.raw.vtt" "C:\path\to\session.vtt"
```

That writes:

- `session.review.json`
- `session.review.md`

The structured review output includes:

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

That keeps the project easy to adapt for another lineage or community without changing the code.
