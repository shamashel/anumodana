"""Glossary loading for fixer and review prompts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GLOSSARY_ROOT = PROJECT_ROOT / "glossaries"

def find_default_glossaries() -> list[Path]:
    """Discover all .txt files in the glossaries directory tree."""
    if not GLOSSARY_ROOT.exists():
        return []
    # Use rglob to recursively find all .txt files.
    return sorted(GLOSSARY_ROOT.rglob("*.txt"))


def build_glossary_paths(
    extra_paths: Iterable[str | Path],
    *,
    include_defaults: bool = True,
) -> list[Path]:
    glossary_paths = [] if not include_defaults else find_default_glossaries()
    glossary_paths.extend(Path(path).expanduser().resolve() for path in extra_paths)
    return glossary_paths


def load_glossary_lines(glossary_paths: list[Path]) -> list[str]:
    lines: list[str] = []
    for path in glossary_paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines
