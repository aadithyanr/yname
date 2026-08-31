#!/usr/bin/env python3
"""Evaluate idea-model coverage, novelty, and output validity."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from generate_ideas import DEFAULT_MODEL
from idea_model import IdeaLanguageModel, NoveltyIndex, generate_ideas, normalize_idea, tokenize
from train_ideas import DEFAULT_CORPUS, load_corpus


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "idea_evaluation.json"


def stratified_split(
    rows: list[dict[str, str]], seed: int, test_fraction: float = 0.2
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row["industry"]].append(row)
    random_source = random.Random(seed)
    train: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for category in sorted(by_category):
        group = by_category[category]
        random_source.shuffle(group)
        test_size = max(1, round(len(group) * test_fraction))
        test.extend(group[:test_size])
        train.extend(group[test_size:])
    return train, test


def heldout_coverage(model: IdeaLanguageModel, rows: list[dict[str, str]]) -> dict[str, float]:
    vocabulary = set(model.global_counts.get((), {}))
    contexts = model.global_counts
    tokens_seen = 0
    tokens_total = 0
    contexts_seen = 0
    contexts_total = 0
    for row in rows:
        tokens = tokenize(row["idea"])
        tokens_seen += sum(token in vocabulary for token in tokens)
        tokens_total += len(tokens)
        padded = ["<START>"] * (model.order - 1) + tokens
        for position in range(model.order - 1, len(padded)):
            context = tuple(padded[position - model.order + 1 : position])
            contexts_seen += context in contexts
            contexts_total += 1
    return {
        "token_coverage": round(tokens_seen / max(1, tokens_total), 4),
        "full_context_coverage": round(contexts_seen / max(1, contexts_total), 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count-per-category", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_corpus(args.data)
    train_rows, test_rows = stratified_split(rows, args.seed)
    evaluation_model = IdeaLanguageModel.train(train_rows)
    full_model = IdeaLanguageModel.load(args.model)
    full_index = NoveltyIndex(row["idea"] for row in rows)

    samples = []
    for category_index, category in enumerate(full_model.categories):
        generated = generate_ideas(
            full_model,
            count=args.count_per_category,
            seed=args.seed + category_index,
            category=category,
        )
        for result in generated:
            nearest, similarity = full_index.nearest(result.idea)
            samples.append(
                {
                    "category": category,
                    "idea": result.idea,
                    "nearest_known_idea": nearest or None,
                    "similarity": round(similarity, 4),
                    "word_count": len(normalize_idea(result.idea).split()),
                }
            )

    similarities = [sample["similarity"] for sample in samples]
    word_counts = [sample["word_count"] for sample in samples]
    requested = len(full_model.categories) * args.count_per_category
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "training_rows": len(train_rows),
        "heldout_rows": len(test_rows),
        "coverage": heldout_coverage(evaluation_model, test_rows),
        "generation": {
            "requested": requested,
            "accepted": len(samples),
            "acceptance_rate": round(len(samples) / max(1, requested), 4),
            "exact_match_count": sum(full_index.contains(sample["idea"]) for sample in samples),
            "mean_nearest_similarity": (
                round(statistics.fmean(similarities), 4) if similarities else None
            ),
            "maximum_nearest_similarity": round(max(similarities), 4) if similarities else None,
            "mean_word_count": round(statistics.fmean(word_counts), 2) if word_counts else None,
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    print(f"Wrote {len(samples)} evaluated samples to {args.output}")


if __name__ == "__main__":
    main()
