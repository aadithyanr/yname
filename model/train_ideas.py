#!/usr/bin/env python3
"""Train the local token-level startup idea model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from idea_model import IdeaLanguageModel


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
DEFAULT_CORPUS = REPOSITORY_ROOT / "data" / "yc_idea_training_corpus.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "yc_idea_model.json.gz"


def load_corpus(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {"idea", "industry"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain the columns: {', '.join(sorted(required))}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--min-context-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = IdeaLanguageModel.train(
        load_corpus(args.data),
        order=args.order,
        min_context_count=args.min_context_count,
    )
    resolved_source = args.data.resolve()
    try:
        source_label = resolved_source.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        source_label = str(args.data)
    model.metadata.update(
        {
            "source": source_label,
            "model_kind": "industry-conditioned token n-gram",
        }
    )
    model.save(args.output)
    print(
        f"Saved {model.metadata['training_rows']:,}-idea model to {args.output} "
        f"with {model.metadata['vocabulary_size']:,} tokens and "
        f"{len(model.categories)} industries."
    )


if __name__ == "__main__":
    main()
