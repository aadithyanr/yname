#!/usr/bin/env python3
"""Generate novel startup names from the locally trained YC character MLPs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def load_trainer_module():
    path = SCRIPT_DIR / "train.py"
    spec = importlib.util.spec_from_file_location("yc_name_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("plain", "conditional"), default="plain"
    )
    parser.add_argument(
        "--category",
        help="Industry for the conditional model; run --list-categories to inspect choices.",
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--creativity", choices=("low", "medium", "high"), default="medium"
    )
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = load_trainer_module()
    repository_root = SCRIPT_DIR.parent
    model_path = SCRIPT_DIR / "artifacts" / f"yc_name_{args.model}_model.npz"
    corpus_path = repository_root / "data" / "yc_name_training_corpus.csv"
    archive = np.load(model_path, allow_pickle=False)

    tokens = archive["tokens"].tolist()
    categories = archive["categories"].tolist()
    if args.list_categories:
        print("\n".join(categories))
        return

    metadata = json.loads(str(archive["metadata_json"]))
    architecture = metadata["architecture"]
    token_to_id = {token: index for index, token in enumerate(tokens)}
    category_to_id = {category: index for index, category in enumerate(categories)}

    if args.model == "conditional":
        if not args.category:
            raise SystemExit(
                "--category is required for the conditional model. "
                "Use --list-categories to see the available values."
            )
        if args.category not in category_to_id:
            raise SystemExit(
                f"Unknown category {args.category!r}. Choices: {', '.join(categories)}"
            )
        category_id = category_to_id[args.category]
    else:
        category_id = 0

    model = trainer.CharacterMLP(
        vocab_size=architecture["vocab_size"],
        category_count=architecture["category_count"],
        context_size=architecture["context_size"],
        embedding_size=architecture["embedding_size"],
        hidden_size=architecture["hidden_size"],
        category_embedding_size=architecture["category_embedding_size"],
        seed=0,
    )
    model.parameters = {
        name: archive[name].copy() for name in model.parameters
    }

    with corpus_path.open(encoding="utf-8", newline="") as handle:
        known_names = {row["name"].casefold() for row in csv.DictReader(handle)}

    temperature_options = {
        "low": (0.66, 0.72, 0.78, 0.84),
        "medium": (0.72, 0.84, 0.96, 1.08),
        "high": (0.88, 1.00, 1.12, 1.24),
    }
    candidates = trainer.generate_candidates(
        model=model,
        token_to_id=token_to_id,
        tokens=tokens,
        category_id=category_id,
        known_names=known_names,
        count=max(1, args.count),
        seed=args.seed,
        temperatures=temperature_options[args.creativity],
    )

    output = [
        {
            **candidate,
            "name": trainer.display_name(candidate["name"]),
            "model": args.model,
            "category": args.category if args.model == "conditional" else None,
        }
        for candidate in candidates
    ]
    if args.as_json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        for candidate in output:
            print(candidate["name"])


if __name__ == "__main__":
    main()
