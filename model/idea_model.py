"""A tiny industry-conditioned token model for generating startup one-liners."""

from __future__ import annotations

import difflib
import gzip
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


START = "<START>"
END = "<END>"
MODEL_VERSION = 1
CONTEXT_SEPARATOR = "\x1f"
TOKEN_RE = re.compile(
    r"[a-z0-9]+(?:['-][a-z0-9]+)*|[&+/]|[.,!?;:()]",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[a-z0-9]")
SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([.,!?;:)])")
SPACE_AFTER_OPEN_RE = re.compile(r"([(])\s+")
MULTISPACE_RE = re.compile(r"\s+")
INITIALISMS = {
    "ai": "AI",
    "api": "API",
    "apis": "APIs",
    "b2b": "B2B",
    "b2c": "B2C",
    "crm": "CRM",
    "erp": "ERP",
    "hd": "HD",
    "hr": "HR",
    "ide": "IDE",
    "iot": "IoT",
    "it": "IT",
    "llm": "LLM",
    "llms": "LLMs",
    "ml": "ML",
    "pr": "PR",
    "vr": "VR",
    "ar": "AR",
    "saas": "SaaS",
    "smb": "SMB",
    "smbs": "SMBs",
}
GENERIC_SOLUTIONS = {
    "a better way",
    "a new way",
    "the best way",
    "the easiest way",
    "the fastest way",
    "the future",
    "we help",
}
CREATIVITY = {
    "low": {"temperature": 0.72, "top_k": 18, "template_top_k": 80},
    "medium": {"temperature": 0.92, "top_k": 42, "template_top_k": 500},
    "high": {"temperature": 1.16, "top_k": None, "template_top_k": None},
}


def tokenize(text: str) -> list[str]:
    """Tokenize a short company one-liner into browser-friendly units."""
    normalized = (
        text.casefold()
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    return TOKEN_RE.findall(normalized)


def normalize_idea(text: str) -> str:
    """Create a comparison key that ignores casing and punctuation."""
    return " ".join(token for token in tokenize(text) if WORD_RE.search(token))


def display_idea(tokens: Sequence[str]) -> str:
    """Turn model tokens into a clean sentence-like one-liner."""
    text = " ".join(tokens)
    text = SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", text)
    text = SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    text = re.sub(r"\s*([/+])\s*", r"\1", text)
    text = MULTISPACE_RE.sub(" ", text).strip(" ,;:-")
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    for source, replacement in INITIALISMS.items():
        text = re.sub(rf"\b{re.escape(source)}\b", replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\br\s*&\s*d\b", "R&D", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d),\s+(?=\d{3}\b)", ",", text)
    text = re.sub(r"(?<=\d)\+\s*", "+ ", text)
    if text[-1] not in ".!?":
        text += "."
    return text


def _extract_for_template(tokens: Sequence[str]) -> tuple[str, str] | None:
    """Extract reusable solution/audience fragments from a `... for ...` idea."""
    try:
        split = tokens.index("for")
    except ValueError:
        return None
    left = list(tokens[:split])
    right = list(tokens[split + 1 :])
    while left and not WORD_RE.search(left[-1]):
        left.pop()
    while right and not WORD_RE.search(right[-1]):
        right.pop()
    while right and not WORD_RE.search(right[0]):
        right.pop(0)
    left_words = sum(bool(WORD_RE.search(token)) for token in left)
    right_words = sum(bool(WORD_RE.search(token)) for token in right)
    if not 2 <= left_words <= 11 or not 2 <= right_words <= 11:
        return None
    if any(token in ".!?" for token in (*left, *right)):
        return None
    if any(token in {"we", "our", "i", "is", "are", "was", "were"} for token in left):
        return None
    if "for" in right:
        return None
    solution = " ".join(left)
    audience = " ".join(right)
    if normalize_idea(solution) in GENERIC_SOLUTIONS:
        return None
    return solution, audience


def _word_shingles(text: str, size: int = 3) -> frozenset[tuple[str, ...]]:
    words = normalize_idea(text).split()
    if not words:
        return frozenset()
    width = min(size, len(words))
    return frozenset(
        tuple(words[index : index + width])
        for index in range(len(words) - width + 1)
    )


def idea_similarity(left: str, right: str) -> float:
    """Estimate lexical similarity using word shingles and a sequence fallback."""
    left_key = normalize_idea(left)
    right_key = normalize_idea(right)
    if left_key == right_key:
        return 1.0
    left_shingles = _word_shingles(left_key)
    right_shingles = _word_shingles(right_key)
    union = left_shingles | right_shingles
    jaccard = len(left_shingles & right_shingles) / len(union) if union else 0.0
    sequence = difflib.SequenceMatcher(None, left_key, right_key).ratio()
    # Sequence similarity catches lightly edited copies; shingle similarity catches
    # copied phrases whose prefixes or suffixes changed.
    return max(jaccard, sequence * 0.82)


class NoveltyIndex:
    """Find exact and near matches without comparing every known idea."""

    def __init__(self, ideas: Iterable[str]) -> None:
        unique: dict[str, str] = {}
        for idea in ideas:
            key = normalize_idea(idea)
            if key:
                unique.setdefault(key, idea)
        self.ideas = list(unique.values())
        self.keys = set(unique)
        # Bigram candidates are deliberately broader than the trigram score so
        # one-word substitutions in short ideas still reach the similarity test.
        self.shingles = [_word_shingles(idea, size=2) for idea in self.ideas]
        inverted: dict[tuple[str, ...], set[int]] = defaultdict(set)
        for index, shingles in enumerate(self.shingles):
            for shingle in shingles:
                inverted[shingle].add(index)
        self.inverted = dict(inverted)

    def contains(self, idea: str) -> bool:
        return normalize_idea(idea) in self.keys

    def nearest(self, idea: str) -> tuple[str, float]:
        candidate_shingles = _word_shingles(idea, size=2)
        candidates: set[int] = set()
        for shingle in candidate_shingles:
            candidates.update(self.inverted.get(shingle, ()))

        # candidate cannot be a high-similarity copy under the shingle metric.
        # When no two-word phrase overlaps, a full scan is unnecessary: the
        # candidate cannot be a high-similarity copy under the shingle metric.
        # candidate cannot be a high-similarity copy under the shingle metric.
        if not candidates:
            return "", 0.0

        nearest_idea = ""
        nearest_score = 0.0
        for index in candidates:
            score = idea_similarity(idea, self.ideas[index])
            if score > nearest_score:
                nearest_idea = self.ideas[index]
                nearest_score = score
        return nearest_idea, nearest_score


ContextCounts = dict[tuple[str, ...], Counter[str]]
TemplateCounts = dict[str, Counter[str]]


def _new_counts() -> defaultdict[tuple[str, ...], Counter[str]]:
    return defaultdict(Counter)


def _encode_context(context: tuple[str, ...]) -> str:
    return CONTEXT_SEPARATOR.join(context)


def _decode_context(value: str) -> tuple[str, ...]:
    return tuple(value.split(CONTEXT_SEPARATOR)) if value else ()


def _encode_counts(counts: Mapping[tuple[str, ...], Counter[str]]) -> dict[str, list[list]]:
    return {
        _encode_context(context): [
            [token, count]
            for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        ]
        for context, counter in sorted(counts.items(), key=lambda item: item[0])
    }


def _decode_counts(payload: Mapping[str, Sequence[Sequence]]) -> ContextCounts:
    return {
        _decode_context(context): Counter({str(token): int(count) for token, count in values})
        for context, values in payload.items()
    }


def _encode_templates(counts: Mapping[str, Counter[str]]) -> dict[str, list[list]]:
    return {
        kind: [[value, count] for value, count in counter.most_common()]
        for kind, counter in counts.items()
    }


def _decode_templates(payload: Mapping[str, Sequence[Sequence]]) -> TemplateCounts:
    return {
        kind: Counter({str(value): int(count) for value, count in values})
        for kind, values in payload.items()
    }


@dataclass(frozen=True)
class GeneratedIdea:
    idea: str
    nearest_known_idea: str
    similarity: float


class IdeaLanguageModel:
    """Interpolated token n-gram model with optional industry conditioning."""

    def __init__(
        self,
        *,
        order: int,
        min_context_count: int,
        global_counts: ContextCounts,
        category_counts: dict[str, ContextCounts],
        global_templates: TemplateCounts,
        category_templates: dict[str, TemplateCounts],
        known_ideas: Sequence[str],
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if order < 2:
            raise ValueError("order must be at least 2")
        self.order = order
        self.min_context_count = min_context_count
        self.global_counts = global_counts
        self.category_counts = category_counts
        self.global_templates = global_templates
        self.category_templates = category_templates
        self.known_ideas = list(known_ideas)
        self.metadata = dict(metadata or {})

    @property
    def categories(self) -> list[str]:
        return sorted(self.category_counts)

    @classmethod
    def train(
        cls,
        rows: Iterable[Mapping[str, str]],
        *,
        order: int = 3,
        min_context_count: int = 2,
    ) -> "IdeaLanguageModel":
        global_counts = _new_counts()
        category_counts: defaultdict[
            str, defaultdict[tuple[str, ...], Counter[str]]
        ] = defaultdict(_new_counts)
        global_templates: defaultdict[str, Counter[str]] = defaultdict(Counter)
        category_templates: defaultdict[str, defaultdict[str, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        known_ideas: list[str] = []
        token_count = 0
        row_count = 0

        for row in rows:
            idea = str(row.get("idea") or "").strip()
            category = str(row.get("industry") or "Other").strip() or "Other"
            tokens = tokenize(idea)
            if len(tokens) < 2:
                continue
            known_ideas.append(idea)
            row_count += 1
            token_count += len(tokens)
            template = _extract_for_template(tokens)
            if template:
                solution, audience = template
                company_key = normalize_idea(str(row.get("company") or ""))
                if company_key and normalize_idea(solution).startswith(company_key):
                    template = None
            if template:
                solution, audience = template
                global_templates["solutions"][solution] += 1
                global_templates["audiences"][audience] += 1
                category_templates[category]["solutions"][solution] += 1
                category_templates[category]["audiences"][audience] += 1
            sequence = [START] * (order - 1) + tokens + [END]

            for position in range(order - 1, len(sequence)):
                target = sequence[position]
                preceding = sequence[max(0, position - order + 1) : position]
                for width in range(0, min(order - 1, len(preceding)) + 1):
                    context = tuple(preceding[-width:]) if width else ()
                    global_counts[context][target] += 1
                    category_counts[category][context][target] += 1

        if not known_ideas:
            raise ValueError("No usable ideas were found in the training rows")

        return cls(
            order=order,
            min_context_count=min_context_count,
            global_counts=dict(global_counts),
            category_counts={
                category: dict(counts) for category, counts in category_counts.items()
            },
            global_templates=dict(global_templates),
            category_templates={
                category: dict(counts) for category, counts in category_templates.items()
            },
            known_ideas=known_ideas,
            metadata={
                "training_rows": row_count,
                "training_tokens": token_count,
                "vocabulary_size": len(global_counts[()]),
                "template_count": sum(global_templates["solutions"].values()),
            },
        )

    def to_payload(self) -> dict:
        return {
            "version": MODEL_VERSION,
            "architecture": {
                "type": "conditional_token_ngram",
                "order": self.order,
                "min_context_count": self.min_context_count,
            },
            "categories": self.categories,
            "metadata": self.metadata,
            "known_ideas": self.known_ideas,
            "global_counts": _encode_counts(self.global_counts),
            "category_counts": {
                category: _encode_counts(self.category_counts[category])
                for category in self.categories
            },
            "global_templates": _encode_templates(self.global_templates),
            "category_templates": {
                category: _encode_templates(self.category_templates.get(category, {}))
                for category in self.categories
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                    json.dump(
                        self.to_payload(),
                        handle,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )

    @classmethod
    def load(cls, path: Path) -> "IdeaLanguageModel":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != MODEL_VERSION:
            raise ValueError(f"Unsupported idea model version: {payload.get('version')!r}")
        architecture = payload["architecture"]
        return cls(
            order=int(architecture["order"]),
            min_context_count=int(architecture["min_context_count"]),
            global_counts=_decode_counts(payload["global_counts"]),
            category_counts={
                category: _decode_counts(counts)
                for category, counts in payload["category_counts"].items()
            },
            global_templates=_decode_templates(payload.get("global_templates", {})),
            category_templates={
                category: _decode_templates(counts)
                for category, counts in payload.get("category_templates", {}).items()
            },
            known_ideas=payload["known_ideas"],
            metadata=payload.get("metadata", {}),
        )

    def _distribution(self, context: Sequence[str], category: str | None) -> Counter[str]:
        category_model = self.category_counts.get(category or "")
        maximum_width = min(self.order - 1, len(context))
        for width in range(maximum_width, -1, -1):
            key = tuple(context[-width:]) if width else ()
            global_counter = self.global_counts.get(key, Counter())
            category_counter = category_model.get(key, Counter()) if category_model else Counter()
            # Category counts are a subset of global counts, so only the global
            # total determines whether a context has independent support.
            total = sum(global_counter.values())
            required = self.min_context_count if width else 1
            if total < required:
                continue

            combined = Counter(global_counter)
            # Industry evidence receives extra weight while global evidence keeps
            # sparse categories grammatical.
            for token, count in category_counter.items():
                combined[token] += count * 3
            return combined
        return Counter()

    def _template_distribution(self, kind: str, category: str | None) -> Counter[str]:
        combined = Counter(self.global_templates.get(kind, {}))
        category_pool = self.category_templates.get(category or "", {}).get(kind, {})
        for value, count in category_pool.items():
            combined[value] += count * 12
        return combined

    @staticmethod
    def _sample_counter_value(
        counter: Counter[str],
        *,
        random_source: random.Random,
        temperature: float,
        top_k: int | None,
    ) -> str:
        return IdeaLanguageModel._sample_token(
            counter,
            random_source=random_source,
            temperature=temperature,
            top_k=top_k,
            blocked=set(),
        )

    def _sample_template(
        self,
        *,
        category: str | None,
        random_source: random.Random,
        temperature: float,
        top_k: int | None,
    ) -> str:
        solutions = self._template_distribution("solutions", category)
        audiences = self._template_distribution("audiences", category)
        if not solutions or not audiences:
            return ""
        solution = self._sample_counter_value(
            solutions,
            random_source=random_source,
            temperature=temperature,
            top_k=top_k,
        )
        audience = self._sample_counter_value(
            audiences,
            random_source=random_source,
            temperature=temperature,
            top_k=top_k,
        )
        return display_idea([*tokenize(solution), "for", *tokenize(audience)])

    @staticmethod
    def _sample_token(
        distribution: Counter[str],
        *,
        random_source: random.Random,
        temperature: float,
        top_k: int | None,
        blocked: set[str],
    ) -> str:
        candidates = [
            (token, count)
            for token, count in distribution.items()
            if token not in blocked
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        if top_k is not None:
            candidates = candidates[:top_k]
        if not candidates:
            return END

        exponent = 1.0 / max(temperature, 1e-6)
        weights = [math.pow(count, exponent) for _, count in candidates]
        threshold = random_source.random() * sum(weights)
        cumulative = 0.0
        for (token, _), weight in zip(candidates, weights, strict=True):
            cumulative += weight
            if cumulative >= threshold:
                return token
        return candidates[-1][0]

    def sample(
        self,
        *,
        category: str | None,
        random_source: random.Random,
        creativity: str = "medium",
        min_words: int = 5,
        max_words: int = 22,
    ) -> str:
        if creativity not in CREATIVITY:
            raise ValueError(f"Unknown creativity {creativity!r}")
        if category and category not in self.category_counts:
            raise ValueError(f"Unknown category {category!r}")

        settings = CREATIVITY[creativity]
        template_probability = {"low": 1.0, "medium": 1.0, "high": 0.65}[creativity]
        if random_source.random() < template_probability:
            template = self._sample_template(
                category=category,
                random_source=random_source,
                temperature=float(settings["temperature"]),
                top_k=settings["template_top_k"],
            )
            if template:
                return template
        context = [START] * (self.order - 1)
        tokens: list[str] = []
        word_count = 0
        maximum_tokens = max_words + 8

        for _ in range(maximum_tokens):
            distribution = self._distribution(context, category)
            blocked = {START}
            if word_count < min_words:
                blocked.add(END)
            token = self._sample_token(
                distribution,
                random_source=random_source,
                temperature=float(settings["temperature"]),
                top_k=settings["top_k"],
                blocked=blocked,
            )
            if token == END:
                break
            if WORD_RE.search(token):
                word_count += 1
                if word_count > max_words:
                    break
            tokens.append(token)
            context = (context + [token])[-(self.order - 1) :]

        return display_idea(tokens)


def is_well_formed_idea(idea: str, *, min_words: int = 5, max_words: int = 22) -> bool:
    words = normalize_idea(idea).split()
    if not min_words <= len(words) <= max_words:
        return False
    if len(set(words)) / len(words) < 0.45:
        return False
    bigrams = list(zip(words, words[1:]))
    if len(set(bigrams)) != len(bigrams):
        return False
    if re.search(r"\b(\w+)(?:\s+\1){2,}\b", " ".join(words)):
        return False
    if idea.count("(") != idea.count(")"):
        return False
    if re.search(r"[.,!?;:]{2,}|[,;:]\.", idea):
        return False
    if len(re.findall(r"[.!?](?:\s|$)", idea)) > 1:
        return False
    return bool(re.search(r"[a-zA-Z]", idea))


def generate_ideas(
    model: IdeaLanguageModel,
    *,
    count: int,
    seed: int,
    category: str | None = None,
    creativity: str = "medium",
    similarity_limit: float = 0.72,
    diversity_limit: float = 0.66,
) -> list[GeneratedIdea]:
    """Sample a diverse batch and reject copied or malformed one-liners."""
    random_source = random.Random(seed)
    novelty = NoveltyIndex(model.known_ideas)
    results: list[GeneratedIdea] = []
    seen: set[str] = set()

    for _ in range(max(200, count * 500)):
        idea = model.sample(
            category=category,
            random_source=random_source,
            creativity=creativity,
        )
        key = normalize_idea(idea)
        if not key or key in seen or novelty.contains(idea) or not is_well_formed_idea(idea):
            continue
        nearest, similarity = novelty.nearest(idea)
        if similarity >= similarity_limit:
            continue
        if any(idea_similarity(idea, result.idea) >= diversity_limit for result in results):
            continue
        seen.add(key)
        results.append(
            GeneratedIdea(
                idea=idea,
                nearest_known_idea=nearest,
                similarity=round(similarity, 4),
            )
        )
        if len(results) >= count:
            break
    return results
