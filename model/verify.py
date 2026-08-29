#!/usr/bin/env python3
"""Verify the checked-in model archives and report their parameter counts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
EXPECTED_PARAMETERS = {"plain": 46_700, "conditional": 49_068}


def main() -> None:
    for kind, expected in EXPECTED_PARAMETERS.items():
        archive = np.load(ARTIFACTS / f"yc_name_{kind}_model.npz", allow_pickle=False)
        metadata = json.loads(str(archive["metadata_json"]))
        actual = int(metadata["parameter_count"])
        if actual != expected:
            raise SystemExit(f"{kind}: expected {expected:,} parameters, found {actual:,}")
        print(f"{kind}: {actual:,} parameters · test loss {metadata['test_loss']:.4f}")


if __name__ == "__main__":
    main()
