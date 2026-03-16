from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GLOSSARY_ROOT = PROJECT_ROOT / "glossaries"

DEFAULT_GLOSSARY_FILES = [
    GLOSSARY_ROOT / "core_chants.txt",
    GLOSSARY_ROOT / "core_theravada_terms.txt",
    GLOSSARY_ROOT / "lineages" / "ajahn_chah.txt",
    GLOSSARY_ROOT / "local_teachers_and_places.txt",
]


def build_glossary_paths(
    extra_paths: Iterable[str | Path],
    *,
    include_defaults: bool = True,
) -> list[Path]:
    glossary_paths = [] if not include_defaults else list(DEFAULT_GLOSSARY_FILES)
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
