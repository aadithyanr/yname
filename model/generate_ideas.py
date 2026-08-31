#!/usr/bin/env python3
"""Generate novel startup ideas with the locally trained token model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idea_model import CREATIVITY, IdeaLanguageModel, generate_ideas


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "artifacts" / "yc_idea_model.json.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--category", help="YC industry used to condition generation.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--creativity", choices=tuple(CREATIVITY), default="medium")
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = IdeaLanguageModel.load(args.model)
    if args.list_categories:
        print("\n".join(model.categories))
        return
    if args.category and args.category not in model.categories:
        raise SystemExit(
            f"Unknown category {args.category!r}. Choices: {', '.join(model.categories)}"
        )

    results = generate_ideas(
        model,
        count=max(1, args.count),
        seed=args.seed,
        category=args.category,
        creativity=args.creativity,
    )
    if len(results) < args.count:
        raise SystemExit(
            f"Only generated {len(results)} acceptable ideas out of {args.count}; "
            "try another seed or a higher creativity setting."
        )

    if args.as_json:
        print(
            json.dumps(
                [
                    {
                        "idea": result.idea,
                        "category": args.category,
                        "creativity": args.creativity,
                        "nearest_known_idea": result.nearest_known_idea or None,
                        "similarity": result.similarity,
                    }
                    for result in results
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        for result in results:
            print(result.idea)


if __name__ == "__main__":
    main()
