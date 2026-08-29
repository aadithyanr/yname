#!/usr/bin/env python3
"""Train tiny, dependency-free character MLPs on the public YC directory.

This trains two deliberately small models:

1. A plain next-character MLP that only sees the previous characters.
2. A conditional MLP that additionally sees the company's broad YC industry.

No language model API or pretrained model is used.  The only runtime dependency is
NumPy.  Model weights are saved as compressed ``.npz`` files so they can later be
exported to ONNX or loaded directly by a small browser implementation.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


START = "<START>"
END = "<END>"
ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789 &'-./")
GENERIC_DESCRIPTORS = {
    "ai",
    "bio",
    "bioscience",
    "biosciences",
    "health",
    "industries",
    "industry",
    "labs",
    "medical",
    "robotics",
    "systems",
    "technologies",
    "technology",
}
BROAD_CATEGORY_MAP = {
    # These labels have only 44 and 18 source records respectively; keeping them
    # separate produces a memorization-prone category rather than useful control.
    "Government": "Other",
    "Unspecified": "Other",
}
LEGAL_SUFFIX_RE = re.compile(
    r"(?:,?\s+(?:inc\.?|incorporated|llc|ltd\.?|limited|pbc|corp\.?|corporation))+$",
    re.IGNORECASE,
)


def normalize_name(raw: str) -> str:
    """Turn a directory display label into the brand text we want to model."""
    text = unicodedata.normalize("NFKD", raw)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("–", "-").replace("—", "-").replace("\xa0", " ")
    # Parenthetical suffixes in the directory overwhelmingly describe renames,
    # acquisitions, or legal entities rather than the current brand.
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = LEGAL_SUFFIX_RE.sub("", text)
    text = text.lower().replace("|", " ")
    text = "".join(ch for ch in text if ch in ALLOWED_CHARS)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([./'-])\s*", r"\1", text)
    return text.strip(" .,/!_-'")


def canonical_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


@dataclass(frozen=True)
class CompanyName:
    name: str
    industry: str
    original: str


def load_companies(path: Path) -> tuple[list[CompanyName], dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    dropped = 0
    changed = 0

    for record in records:
        original = str(record.get("name") or "").strip()
        name = normalize_name(original)
        if len(canonical_key(name)) < 2:
            dropped += 1
            continue
        if name != original.casefold():
            changed += 1
        industry = str(record.get("industry") or "Unspecified").strip()
        industry = BROAD_CATEGORY_MAP.get(industry, industry)
        grouped[name.casefold()].append((name, industry, original))

    companies: list[CompanyName] = []
    duplicate_rows = 0
    conflicting_categories = 0
    for rows in grouped.values():
        duplicate_rows += len(rows) - 1
        counts = Counter(row[1] for row in rows)
        if len(counts) > 1:
            conflicting_categories += 1
        # Majority category; alphabetical tie-break makes this deterministic.
        industry = sorted(counts, key=lambda value: (-counts[value], value))[0]
        name, _, original = rows[0]
        companies.append(CompanyName(name=name, industry=industry, original=original))

    companies.sort(key=lambda company: company.name)
    stats = {
        "source_records": len(records),
        "usable_unique_names": len(companies),
        "duplicate_rows_removed_after_cleaning": duplicate_rows,
        "names_changed_by_cleaning": changed,
        "rows_dropped": dropped,
        "duplicate_names_with_conflicting_categories": conflicting_categories,
    }
    return companies, stats


def stratified_split(
    companies: list[CompanyName], seed: int
) -> tuple[list[CompanyName], list[CompanyName], list[CompanyName]]:
    """Create deterministic 80/10/10 splits, stratified by industry."""
    rng = np.random.default_rng(seed)
    by_industry: dict[str, list[CompanyName]] = defaultdict(list)
    for company in companies:
        by_industry[company.industry].append(company)

    train: list[CompanyName] = []
    validation: list[CompanyName] = []
    test: list[CompanyName] = []
    for industry in sorted(by_industry):
        group = by_industry[industry]
        indices = rng.permutation(len(group))
        n_test = max(1, round(len(group) * 0.10))
        n_validation = max(1, round(len(group) * 0.10))
        test.extend(group[index] for index in indices[:n_test])
        validation.extend(
            group[index] for index in indices[n_test : n_test + n_validation]
        )
        train.extend(group[index] for index in indices[n_test + n_validation :])
    return train, validation, test


def build_vocabulary(companies: Iterable[CompanyName]) -> tuple[list[str], dict[str, int]]:
    chars = sorted({char for company in companies for char in company.name})
    tokens = [START, END, *chars]
    return tokens, {token: index for index, token in enumerate(tokens)}


def make_examples(
    companies: Iterable[CompanyName],
    token_to_id: dict[str, int],
    category_to_id: dict[str, int],
    context_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contexts: list[list[int]] = []
    targets: list[int] = []
    categories: list[int] = []
    start_id = token_to_id[START]
    end_id = token_to_id[END]

    for company in companies:
        context = [start_id] * context_size
        for char in company.name:
            contexts.append(context.copy())
            targets.append(token_to_id[char])
            categories.append(category_to_id[company.industry])
            context = context[1:] + [token_to_id[char]]
        contexts.append(context.copy())
        targets.append(end_id)
        categories.append(category_to_id[company.industry])

    return (
        np.asarray(contexts, dtype=np.int32),
        np.asarray(targets, dtype=np.int32),
        np.asarray(categories, dtype=np.int32),
    )


class CharacterMLP:
    """Embedding -> flatten (+ category) -> tanh -> logits."""

    def __init__(
        self,
        vocab_size: int,
        category_count: int,
        context_size: int,
        embedding_size: int,
        hidden_size: int,
        category_embedding_size: int,
        seed: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.category_count = category_count
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.category_embedding_size = category_embedding_size
        self.conditional = category_embedding_size > 0
        rng = np.random.default_rng(seed)

        input_size = context_size * embedding_size + category_embedding_size
        self.parameters: dict[str, np.ndarray] = {
            "char_embedding": rng.normal(
                0.0, 0.10, (vocab_size, embedding_size)
            ).astype(np.float32),
            "w1": rng.normal(0.0, 1.0 / math.sqrt(input_size), (input_size, hidden_size)).astype(
                np.float32
            ),
            "b1": np.zeros(hidden_size, dtype=np.float32),
            "w2": rng.normal(0.0, 0.01, (hidden_size, vocab_size)).astype(np.float32),
            "b2": np.zeros(vocab_size, dtype=np.float32),
        }
        if self.conditional:
            self.parameters["category_embedding"] = rng.normal(
                0.0, 0.10, (category_count, category_embedding_size)
            ).astype(np.float32)
            # This direct category-to-character path makes the conditioning harder
            # for the hidden layer to ignore, particularly around word endings.
            self.parameters["category_output"] = np.zeros(
                (category_count, vocab_size), dtype=np.float32
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.size for parameter in self.parameters.values())

    def _features(
        self, contexts: np.ndarray, categories: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        embedded = self.parameters["char_embedding"][contexts]
        flattened = embedded.reshape(len(contexts), -1)
        if self.conditional:
            category_features = self.parameters["category_embedding"][categories]
            features = np.concatenate((flattened, category_features), axis=1)
        else:
            features = flattened
        return embedded, features

    def logits(self, contexts: np.ndarray, categories: np.ndarray) -> np.ndarray:
        _, features = self._features(contexts, categories)
        hidden = np.tanh(features @ self.parameters["w1"] + self.parameters["b1"])
        logits = hidden @ self.parameters["w2"] + self.parameters["b2"]
        if self.conditional:
            logits += self.parameters["category_output"][categories]
        return logits

    def loss_and_gradients(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        categories: np.ndarray,
        weight_decay: float,
    ) -> tuple[float, dict[str, np.ndarray]]:
        _, features = self._features(contexts, categories)
        hidden = np.tanh(features @ self.parameters["w1"] + self.parameters["b1"])
        logits = hidden @ self.parameters["w2"] + self.parameters["b2"]
        if self.conditional:
            logits += self.parameters["category_output"][categories]

        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        row = np.arange(len(targets))
        loss = float(-np.log(probabilities[row, targets] + 1e-12).mean())

        d_logits = probabilities
        d_logits[row, targets] -= 1.0
        d_logits /= len(targets)

        gradients: dict[str, np.ndarray] = {}
        gradients["w2"] = hidden.T @ d_logits + weight_decay * self.parameters["w2"]
        gradients["b2"] = d_logits.sum(axis=0)
        d_hidden = d_logits @ self.parameters["w2"].T
        d_pre_hidden = d_hidden * (1.0 - hidden * hidden)
        gradients["w1"] = features.T @ d_pre_hidden + weight_decay * self.parameters["w1"]
        gradients["b1"] = d_pre_hidden.sum(axis=0)
        d_features = d_pre_hidden @ self.parameters["w1"].T

        char_feature_width = self.context_size * self.embedding_size
        d_char = d_features[:, :char_feature_width].reshape(
            len(contexts), self.context_size, self.embedding_size
        )
        d_embedding = np.zeros_like(self.parameters["char_embedding"])
        np.add.at(d_embedding, contexts.reshape(-1), d_char.reshape(-1, self.embedding_size))
        gradients["char_embedding"] = d_embedding

        if self.conditional:
            d_category = np.zeros_like(self.parameters["category_embedding"])
            np.add.at(d_category, categories, d_features[:, char_feature_width:])
            gradients["category_embedding"] = d_category
            d_category_output = np.zeros_like(self.parameters["category_output"])
            np.add.at(d_category_output, categories, d_logits)
            gradients["category_output"] = d_category_output

        return loss, gradients


class Adam:
    def __init__(
        self, parameters: dict[str, np.ndarray], learning_rate: float
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.step_number = 0
        self.m = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.v = {name: np.zeros_like(value) for name, value in parameters.items()}

    def step(
        self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray]
    ) -> None:
        self.step_number += 1
        correction1 = 1.0 - self.beta1**self.step_number
        correction2 = 1.0 - self.beta2**self.step_number
        for name, parameter in parameters.items():
            gradient = gradients[name]
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * gradient
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * gradient * gradient
            m_hat = self.m[name] / correction1
            v_hat = self.v[name] / correction2
            parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def evaluate(
    model: CharacterMLP,
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
    chunk_size: int = 4096,
) -> float:
    contexts, targets, categories = data
    total_loss = 0.0
    for start in range(0, len(targets), chunk_size):
        end = start + chunk_size
        logits = model.logits(contexts[start:end], categories[start:end])
        logits -= logits.max(axis=1, keepdims=True)
        log_normalizer = np.log(np.exp(logits).sum(axis=1))
        correct = logits[np.arange(len(logits)), targets[start:end]]
        total_loss += float((log_normalizer - correct).sum())
    return total_loss / len(targets)


def train_model(
    label: str,
    model: CharacterMLP,
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    eval_every: int,
    patience: int,
    seed: int,
    category_balance_power: float = 0.0,
) -> tuple[list[dict], int]:
    rng = np.random.default_rng(seed)
    optimizer = Adam(model.parameters, learning_rate)
    history: list[dict] = []
    best_loss = float("inf")
    best_parameters: dict[str, np.ndarray] | None = None
    stale_evaluations = 0
    started = time.perf_counter()

    category_buckets: dict[int, np.ndarray] = {}
    category_ids = np.unique(train_data[2])
    category_probabilities: np.ndarray | None = None
    if category_balance_power > 0.0:
        for category_id in category_ids:
            category_buckets[int(category_id)] = np.flatnonzero(
                train_data[2] == category_id
            )
        category_sizes = np.asarray(
            [len(category_buckets[int(category_id)]) for category_id in category_ids],
            dtype=np.float64,
        )
        category_probabilities = category_sizes ** (1.0 - category_balance_power)
        category_probabilities /= category_probabilities.sum()

    print(f"\nTraining {label}: {model.parameter_count:,} parameters")
    for step in range(1, steps + 1):
        if category_probabilities is None:
            indices = rng.integers(0, len(train_data[1]), size=batch_size)
        else:
            sampled_categories = rng.choice(
                category_ids, size=batch_size, p=category_probabilities
            )
            indices = np.empty(batch_size, dtype=np.int64)
            for category_id in category_ids:
                positions = np.flatnonzero(sampled_categories == category_id)
                if len(positions):
                    bucket = category_buckets[int(category_id)]
                    indices[positions] = rng.choice(bucket, size=len(positions))
        batch = tuple(array[indices] for array in train_data)
        training_loss, gradients = model.loss_and_gradients(
            batch[0], batch[1], batch[2], weight_decay
        )
        optimizer.step(model.parameters, gradients)

        if step == 1 or step % eval_every == 0 or step == steps:
            validation_loss = evaluate(model, validation_data)
            elapsed = time.perf_counter() - started
            item = {
                "step": step,
                "batch_loss": training_loss,
                "validation_loss": validation_loss,
                "elapsed_seconds": elapsed,
            }
            history.append(item)
            print(
                f"  step {step:5d} | batch {training_loss:.4f} | "
                f"validation {validation_loss:.4f} | {elapsed:.1f}s"
            )

            if validation_loss < best_loss - 0.001:
                best_loss = validation_loss
                best_parameters = {
                    name: value.copy() for name, value in model.parameters.items()
                }
                stale_evaluations = 0
            else:
                stale_evaluations += 1
                if stale_evaluations >= patience:
                    print(f"  early stopping at step {step}")
                    break

    if best_parameters is not None:
        model.parameters = best_parameters
    best_step = min(history, key=lambda item: item["validation_loss"])["step"]
    return history, int(best_step)


def next_probabilities(
    model: CharacterMLP,
    context: list[int],
    category_id: int,
    temperature: float,
) -> np.ndarray:
    contexts = np.asarray([context], dtype=np.int32)
    categories = np.asarray([category_id], dtype=np.int32)
    logits = model.logits(contexts, categories)[0] / temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return probabilities


def sample_name(
    model: CharacterMLP,
    token_to_id: dict[str, int],
    tokens: list[str],
    category_id: int,
    rng: np.random.Generator,
    temperature: float,
    min_length: int = 3,
    max_length: int = 22,
) -> tuple[str, float]:
    context = [token_to_id[START]] * model.context_size
    end_id = token_to_id[END]
    start_id = token_to_id[START]
    chars: list[str] = []
    log_probability = 0.0

    for _ in range(max_length):
        probabilities = next_probabilities(
            model, context, category_id, temperature
        )
        probabilities[start_id] = 0.0
        if len(chars) < min_length:
            probabilities[end_id] = 0.0
        probabilities /= probabilities.sum()
        token_id = int(rng.choice(len(tokens), p=probabilities))
        log_probability += math.log(float(probabilities[token_id]) + 1e-12)
        if token_id == end_id:
            break
        chars.append(tokens[token_id])
        context = context[1:] + [token_id]
    return "".join(chars).strip(), log_probability / max(1, len(chars))


def model_score(
    model: CharacterMLP,
    name: str,
    token_to_id: dict[str, int],
    category_id: int,
) -> float:
    """Average untempered log probability, comparable across sampled temperatures."""
    context = [token_to_id[START]] * model.context_size
    log_probability = 0.0
    token_ids = [token_to_id[char] for char in name] + [token_to_id[END]]
    for token_id in token_ids:
        probabilities = next_probabilities(model, context, category_id, 1.0)
        log_probability += math.log(float(probabilities[token_id]) + 1e-12)
        if token_id != token_to_id[END]:
            context = context[1:] + [token_id]
    return log_probability / len(token_ids)


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def descriptive_root(name: str) -> str:
    words = name.casefold().split()
    while len(words) > 1 and words[-1] in GENERIC_DESCRIPTORS:
        words.pop()
    return " ".join(words)


def descriptor_is_misspelled(name: str) -> bool:
    for word in name.casefold().split():
        if word in GENERIC_DESCRIPTORS or len(word) < 4:
            continue
        for descriptor in GENERIC_DESCRIPTORS:
            if abs(len(word) - len(descriptor)) <= 1 and edit_distance(word, descriptor) == 1:
                return True
    return False


def trigrams(name: str) -> set[str]:
    padded = f"^^{canonical_key(name)}$$"
    return {padded[index : index + 3] for index in range(max(0, len(padded) - 2))}


def looks_usable(name: str) -> bool:
    if not (3 <= len(name) <= 20):
        return False
    if not name[0].isalnum() or not name[-1].isalnum():
        return False
    if len(re.findall(r"[a-z]", name)) < 2:
        return False
    if re.search(r"(.)\1\1", name):
        return False
    if re.search(r"[ ./'-]{2}", name):
        return False
    if name.count(" ") > 2:
        return False
    letters = re.sub(r"[^a-z]", "", name)
    if len(letters) >= 5 and not re.search(r"[aeiouy]", letters):
        return False
    if re.search(r"[bcdfghjklmnpqrstvwxz]{6}", letters):
        return False
    if name.casefold() in GENERIC_DESCRIPTORS or descriptor_is_misspelled(name):
        return False
    return True


def generate_candidates(
    model: CharacterMLP,
    token_to_id: dict[str, int],
    tokens: list[str],
    category_id: int,
    known_names: set[str],
    count: int,
    seed: int,
    temperatures: tuple[float, ...] = (0.72, 0.84, 0.96, 1.08),
) -> list[dict]:
    rng = np.random.default_rng(seed)
    candidates: dict[str, dict] = {}
    known_roots = {descriptive_root(name) for name in known_names}
    canonical_to_known: dict[str, str] = {}
    trigram_index: dict[str, set[str]] = defaultdict(set)
    length_index: dict[int, set[str]] = defaultdict(set)
    for known_name in known_names:
        canonical = canonical_key(known_name)
        if not canonical:
            continue
        canonical_to_known.setdefault(canonical, known_name)
        length_index[len(canonical)].add(canonical)
        for gram in trigrams(known_name):
            trigram_index[gram].add(canonical)
    attempts = count * 30
    for attempt in range(attempts):
        temperature = temperatures[attempt % len(temperatures)]
        name, score = sample_name(
            model,
            token_to_id,
            tokens,
            category_id,
            rng,
            temperature,
        )
        key = name.casefold()
        root = descriptive_root(key)
        if (
            key in known_names
            or key in candidates
            or root in known_roots
            or not looks_usable(name)
        ):
            continue

        canonical = canonical_key(name)
        comparison_pool: set[str] = set()
        for gram in trigrams(name):
            comparison_pool.update(trigram_index.get(gram, ()))
        if len(canonical) <= 5:
            for length in range(max(1, len(canonical) - 2), len(canonical) + 3):
                comparison_pool.update(length_index.get(length, ()))
        nearest_canonical = ""
        nearest_similarity = 0.0
        for known_canonical in comparison_pool:
            similarity = difflib.SequenceMatcher(
                None, canonical, known_canonical
            ).ratio()
            if similarity > nearest_similarity:
                nearest_similarity = similarity
                nearest_canonical = known_canonical
        threshold = 0.80 if len(canonical) <= 5 else 0.84
        if nearest_similarity >= threshold:
            continue
        likelihood = model_score(model, name, token_to_id, category_id)
        # Avoid letting highly predictable directory boilerplate monopolize the
        # results while retaining some two-word names for authentic variety.
        descriptor_count = sum(
            word in GENERIC_DESCRIPTORS for word in key.split()
        )
        length_penalty = abs(len(name) - 8) * 0.018
        descriptor_penalty = descriptor_count * 0.28
        multiword_penalty = max(0, len(key.split()) - 1) * 0.06
        candidates[key] = {
            "name": name,
            "score": likelihood
            - length_penalty
            - descriptor_penalty
            - multiword_penalty,
            "model_log_probability": likelihood,
            "nearest_known_name": canonical_to_known.get(nearest_canonical, ""),
            "nearest_known_similarity": nearest_similarity,
            "temperature": temperature,
        }

    # Greedy diversity pass: a page of minor spelling variations is less useful
    # than a slightly wider set of good candidates.
    pool = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
    selected: list[dict] = []
    selected_grams: list[set[str]] = []
    while pool and len(selected) < count:
        best_index = 0
        best_adjusted = -float("inf")
        for index, candidate in enumerate(pool[: max(100, count * 5)]):
            grams = trigrams(candidate["name"])
            similarity = 0.0
            for prior in selected_grams:
                union = grams | prior
                if union:
                    similarity = max(similarity, len(grams & prior) / len(union))
            adjusted = candidate["score"] - 0.35 * similarity
            if adjusted > best_adjusted:
                best_index = index
                best_adjusted = adjusted
        chosen = pool.pop(best_index)
        selected.append(chosen)
        selected_grams.append(trigrams(chosen["name"]))
    return selected


def display_name(name: str) -> str:
    return name[:1].upper() + name[1:]


def save_model(
    path: Path,
    model: CharacterMLP,
    tokens: list[str],
    categories: list[str],
    metadata: dict,
) -> None:
    arrays = dict(model.parameters)
    arrays["tokens"] = np.asarray(tokens)
    arrays["categories"] = np.asarray(categories)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)


def write_training_corpus(path: Path, companies: Iterable[CompanyName]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("name", "industry", "original_name"))
        for company in companies:
            writer.writerow((company.name, company.industry, company.original))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--context-size", type=int, default=10)
    parser.add_argument("--embedding-size", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=160)
    parser.add_argument("--category-embedding-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    companies, cleaning_stats = load_companies(args.data)
    train_companies, validation_companies, test_companies = stratified_split(
        companies, args.seed
    )
    tokens, token_to_id = build_vocabulary(companies)
    categories = sorted({company.industry for company in companies})
    category_to_id = {category: index for index, category in enumerate(categories)}

    train_data = make_examples(
        train_companies, token_to_id, category_to_id, args.context_size
    )
    validation_data = make_examples(
        validation_companies, token_to_id, category_to_id, args.context_size
    )
    test_data = make_examples(
        test_companies, token_to_id, category_to_id, args.context_size
    )

    print("Dataset")
    print(f"  unique cleaned names: {len(companies):,}")
    print(
        f"  split: {len(train_companies):,} train / "
        f"{len(validation_companies):,} validation / {len(test_companies):,} test"
    )
    print(
        f"  examples: {len(train_data[1]):,} train / "
        f"{len(validation_data[1]):,} validation / {len(test_data[1]):,} test"
    )
    print(f"  vocabulary: {len(tokens)} tokens; categories: {len(categories)}")

    common = dict(
        vocab_size=len(tokens),
        category_count=len(categories),
        context_size=args.context_size,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    models = {
        "plain": CharacterMLP(category_embedding_size=0, **common),
        "conditional": CharacterMLP(
            category_embedding_size=args.category_embedding_size, **common
        ),
    }

    report: dict = {
        "seed": args.seed,
        "cleaning": cleaning_stats,
        "split_names": {
            "train": len(train_companies),
            "validation": len(validation_companies),
            "test": len(test_companies),
        },
        "examples": {
            "train": len(train_data[1]),
            "validation": len(validation_data[1]),
            "test": len(test_data[1]),
        },
        "vocabulary": tokens,
        "categories": categories,
        "config": vars(args) | {"data": str(args.data), "output_dir": str(args.output_dir)},
        "models": {},
    }
    known_names = {company.name.casefold() for company in companies}

    for model_index, (label, model) in enumerate(models.items()):
        history, best_step = train_model(
            label,
            model,
            train_data,
            validation_data,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
            patience=args.patience,
            seed=args.seed + 100 + model_index,
            category_balance_power=0.45 if label == "conditional" else 0.0,
        )
        train_loss = evaluate(model, train_data)
        validation_loss = evaluate(model, validation_data)
        test_loss = evaluate(model, test_data)

        sample_groups: dict[str, list[dict]] = {}
        if label == "plain":
            sample_groups["Any"] = generate_candidates(
                model,
                token_to_id,
                tokens,
                category_id=0,
                known_names=known_names,
                count=24,
                seed=args.seed + 1000,
            )
        else:
            for category in categories:
                sample_groups[category] = generate_candidates(
                    model,
                    token_to_id,
                    tokens,
                    category_id=category_to_id[category],
                    known_names=known_names,
                    count=12,
                    seed=args.seed + 2000 + category_to_id[category],
                )

        model_report = {
            "parameter_count": model.parameter_count,
            "architecture": {
                "context_size": model.context_size,
                "embedding_size": model.embedding_size,
                "hidden_size": model.hidden_size,
                "category_embedding_size": model.category_embedding_size,
                "category_count": model.category_count,
                "vocab_size": model.vocab_size,
                "conditional": model.conditional,
            },
            "best_step": best_step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "test_loss": test_loss,
            "test_perplexity": math.exp(test_loss),
            "history": history,
            "samples": sample_groups,
        }
        report["models"][label] = model_report
        save_model(
            args.output_dir / f"yc_name_{label}_model.npz",
            model,
            tokens,
            categories,
            model_report,
        )

    write_training_corpus(args.output_dir / "yc_name_training_corpus.csv", companies)
    (args.output_dir / "yc_name_training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# YC name-model training results",
        "",
        "Two small character MLPs were trained from scratch. No LLM or pretrained model was used.",
        "",
        "| Model | Parameters | Best step | Validation loss | Test loss | Test perplexity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, result in report["models"].items():
        lines.append(
            f"| {label.title()} | {result['parameter_count']:,} | {result['best_step']:,} | "
            f"{result['validation_loss']:.4f} | {result['test_loss']:.4f} | "
            f"{result['test_perplexity']:.2f} |"
        )
    lines.extend(("", "## Plain model samples", ""))
    plain_names = [display_name(item["name"]) for item in report["models"]["plain"]["samples"]["Any"]]
    lines.append(", ".join(plain_names))
    lines.extend(("", "## Category-conditioned samples", ""))
    for category, samples in report["models"]["conditional"]["samples"].items():
        lines.append(f"### {category}")
        lines.append("")
        lines.append(", ".join(display_name(item["name"]) for item in samples))
        lines.append("")
    lines.extend(
        (
            "## Notes",
            "",
            f"- Source records: {cleaning_stats['source_records']:,}",
            f"- Unique cleaned names: {cleaning_stats['usable_unique_names']:,}",
            f"- Training/validation/test names: {len(train_companies):,}/{len(validation_companies):,}/{len(test_companies):,}",
            "- Generated samples shown above exclude exact matches to every known directory name.",
            "- Candidates at or above 0.84 similarity to a known name are also rejected (0.80 for very short names).",
            "- YC's sparse Government and Unspecified labels are merged into Other for conditioning.",
            "- Lower loss is better; perplexity is the model's average effective number of next-character choices.",
        )
    )
    (args.output_dir / "yc_name_training_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\nFinal comparison")
    for label, result in report["models"].items():
        print(
            f"  {label:11s} | validation {result['validation_loss']:.4f} | "
            f"test {result['test_loss']:.4f} | perplexity {result['test_perplexity']:.2f}"
        )
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
