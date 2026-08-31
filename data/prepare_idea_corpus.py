#!/usr/bin/env python3
"""Build a clean startup-idea corpus from the public YC company directory."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import unicodedata
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE = "https://yc-oss.github.io/api/companies/all.json"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "yc_idea_training_corpus.csv"
DEFAULT_REPORT = REPOSITORY_ROOT / "data" / "yc_idea_corpus_report.json"
BROAD_CATEGORY_MAP = {
    "Government": "Other",
    "Unspecified": "Other",
}
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
HISTORICAL_STATUS_RE = re.compile(r"\b(?:acquired by|shut down)\b", re.IGNORECASE)


def normalize_text(raw: object) -> str:
    """Normalize directory text without changing its meaning."""
    text = html.unescape(str(raw or ""))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = URL_RE.sub("", text)
    return WHITESPACE_RE.sub(" ", text).strip(" \t\r\n-")


def canonical_idea(text: str) -> str:
    """Return a stable key for exact idea deduplication."""
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def prepare_records(records: Iterable[dict]) -> tuple[list[dict[str, str]], dict]:
    """Clean, deduplicate, and sort YC company one-liners."""
    source_records = list(records)
    output: list[dict[str, str]] = []
    seen_ideas: set[str] = set()
    dropped_missing = 0
    dropped_invalid = 0
    dropped_truncated = 0
    dropped_historical = 0
    duplicate_ideas = 0

    for record in source_records:
        company = normalize_text(record.get("name"))
        idea = normalize_text(record.get("one_liner"))
        if not company or not idea:
            dropped_missing += 1
            continue
        # The directory snapshot contains hundreds of display-truncated
        # one-liners. Their trailing ellipses teach the generator to stop in the
        # middle of a thought, so exclude them from the training corpus.
        if idea.endswith(("...", "…")):
            dropped_truncated += 1
            continue
        if HISTORICAL_STATUS_RE.search(idea):
            dropped_historical += 1
            continue

        key = canonical_idea(idea)
        if (
            len(key) < 5
            or not re.search(r"[a-z]", key)
            or key == canonical_idea(company)
        ):
            dropped_invalid += 1
            continue
        if key in seen_ideas:
            duplicate_ideas += 1
            continue
        seen_ideas.add(key)

        industry = normalize_text(record.get("industry")) or "Unspecified"
        industry = BROAD_CATEGORY_MAP.get(industry, industry)
        subindustry = normalize_text(record.get("subindustry"))
        raw_tags = record.get("tags")
        tags = raw_tags if isinstance(raw_tags, list) else []
        output.append(
            {
                "company": company,
                "idea": idea,
                "industry": industry,
                "subindustry": subindustry,
                "tags": "|".join(
                    tag for tag in (normalize_text(value) for value in tags) if tag
                ),
            }
        )

    output.sort(key=lambda row: (row["company"].casefold(), row["idea"].casefold()))
    industries = Counter(row["industry"] for row in output)
    stats = {
        "source_records": len(source_records),
        "usable_unique_ideas": len(output),
        "rows_dropped_missing_company_or_idea": dropped_missing,
        "rows_dropped_invalid_idea": dropped_invalid,
        "rows_dropped_truncated_idea": dropped_truncated,
        "rows_dropped_historical_status": dropped_historical,
        "duplicate_ideas_removed": duplicate_ideas,
        "industries": dict(sorted(industries.items())),
    }
    return output, stats


def load_records(source: str) -> list[dict]:
    """Load JSON records from a local path or an HTTPS URL."""
    if source.startswith(("https://", "http://")):
        request = urllib.request.Request(
            source,
            headers={"User-Agent": "yname-idea-corpus/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))

    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("Expected the source to contain a JSON array of objects")
    return payload


def write_corpus(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("company", "idea", "industry", "subindustry", "tags"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Local JSON path or URL (default: the yc-oss public company snapshot).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, stats = prepare_records(load_records(args.source))
    write_corpus(args.output, rows)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        **stats,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(rows):,} unique startup ideas to {args.output} "
        f"({stats['duplicate_ideas_removed']:,} duplicate ideas removed)."
    )


if __name__ == "__main__":
    main()
