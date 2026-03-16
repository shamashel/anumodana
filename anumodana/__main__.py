from __future__ import annotations

import sys
from collections.abc import Sequence

from anumodana import correction, pipeline, review


ROOT_HELP = """\
usage:
  python -m anumodana [batch options]
  python -m anumodana batch [batch options]
  python -m anumodana cleanup <input.vtt> [cleanup options]
  python -m anumodana review <raw.vtt> <cleaned.vtt> [review options]

commands:
  batch     run the full transcription pipeline
  cleanup   run only the subtitle cleanup pass
  review    run only the review pass

notes:
  running without a subcommand defaults to the full batch pipeline
  use `python -m anumodana batch --help` for pipeline options
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help", "help"}:
        print(ROOT_HELP, end="")
        return 0

    if args and args[0] == "batch":
        return pipeline.main(args[1:], prog="python -m anumodana batch")
    if args and args[0] == "cleanup":
        return correction.main(args[1:], prog="python -m anumodana cleanup")
    if args and args[0] == "review":
        return review.main(args[1:], prog="python -m anumodana review")

    return pipeline.main(args, prog="python -m anumodana")


if __name__ == "__main__":
    raise SystemExit(main())
