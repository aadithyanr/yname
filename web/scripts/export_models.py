#!/usr/bin/env python3
"""Export local model archives into compact browser-readable assets."""

from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
MODEL_ROOT = REPOSITORY_ROOT / "model" / "artifacts"
DATA_ROOT = REPOSITORY_ROOT / "data"
MODEL_OUTPUT = PROJECT_ROOT / "public" / "models"


def canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def export_model(kind: str) -> None:
    source = MODEL_ROOT / f"yc_name_{kind}_model.npz"
    archive = np.load(source, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"]))
    architecture = metadata["architecture"]
    parameter_names = ["char_embedding", "w1", "b1", "w2", "b2"]
    if architecture["conditional"]:
        parameter_names.extend(("category_embedding", "category_output"))

    chunks: list[bytes] = []
    parameters: dict[str, dict] = {}
    byte_offset = 0
    for name in parameter_names:
        array = np.ascontiguousarray(archive[name], dtype="<f4")
        payload = array.tobytes(order="C")
        parameters[name] = {
            "shape": list(array.shape),
            "byteOffset": byte_offset,
            "length": int(array.size),
        }
        chunks.append(payload)
        byte_offset += len(payload)

    manifest = {
        "version": 1,
        "kind": kind,
        "architecture": architecture,
        "tokens": archive["tokens"].tolist(),
        "categories": archive["categories"].tolist(),
        "parameters": parameters,
        "training": {
            "sourceRecords": 6194,
            "uniqueCleanedNames": 6090,
            "bestStep": metadata["best_step"],
            "testLoss": metadata["test_loss"],
        },
    }
    (MODEL_OUTPUT / f"{kind}.bin").write_bytes(b"".join(chunks))
    (MODEL_OUTPUT / f"{kind}.json").write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )


def export_known_names() -> None:
    source = DATA_ROOT / "yc_name_training_corpus.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = sorted({row["name"].casefold() for row in rows})
    canonical_names = sorted({canonical_key(name) for name in names if canonical_key(name)})
    (MODEL_OUTPUT / "known-names.json").write_text(
        json.dumps(
            {"names": names, "canonicalNames": canonical_names},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def export_idea_model() -> None:
    """Export the compact phrase sampler used by the browser idea mode."""
    source = MODEL_ROOT / "yc_idea_model.json.gz"
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    browser_payload = {
        "version": payload["version"],
        "categories": payload["categories"],
        "knownIdeas": payload["known_ideas"],
        "globalTemplates": payload["global_templates"],
        "categoryTemplates": payload["category_templates"],
    }
    (MODEL_OUTPUT / "ideas.json").write_text(
        json.dumps(browser_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def main() -> None:
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    export_model("plain")
    export_model("conditional")
    export_known_names()
    export_idea_model()
    print(f"exported browser assets to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
