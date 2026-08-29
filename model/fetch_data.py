#!/usr/bin/env python3
"""Download the latest public YC company directory snapshot."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


SOURCE_URL = "https://yc-oss.github.io/api/companies/all.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "yc_companies.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        args.url,
        headers={"User-Agent": "yname-dataset-fetcher/1.0"},
    )
    with urllib.request.urlopen(request) as response:
        payload = response.read()
    args.output.write_bytes(payload)
    print(f"downloaded {len(payload):,} bytes to {args.output}")


if __name__ == "__main__":
    main()
