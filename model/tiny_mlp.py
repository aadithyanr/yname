#!/usr/bin/env python3
"""A tiny, educational startup-name MLP implemented directly in NumPy.

This is deliberately *not* the production trainer. It uses a handful of
recognizable YC company names and exposes all the important moving parts:

1. Turn each name into "previous characters -> next character" examples.
2. Embed the previous characters as learned vectors.
3. Pass those vectors through a tanh hidden layer.
4. Predict a probability for every possible next character.
5. Measure cross-entropy loss and update every parameter with backprop.
6. Sample the trained model one character at a time.

Run:
    python tiny_yc_name_mlp.py

Optional:
    python tiny_yc_name_mlp.py --steps 100 --seed 7 --count 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


# A deliberately small teaching corpus. Names are lowercased during training,
# so the model focuses on spelling patterns rather than capitalization rules.
RAW_NAMES = [
    "Reddit",
    "Dropbox",
    "Airbnb",
    "Stripe",
    "Segment",
    "Retool",
    "Supabase",
    "OpenAI",
    "Context.dev",
    "Instacart",
    "Coinbase",
    "Vercel",
]

CONTEXT_SIZE = 8
EMBEDDING_SIZE = 16
HIDDEN_SIZE = 64
MIN_NAME_LENGTH = 3
MAX_NAME_LENGTH = 20

# START pads the empty history. END tells generation when to stop. They are
# separate from real punctuation, so a period remains valid in Context.dev.
START = "<START>"
END = "<END>"


def normalize_name(name: str) -> str:
    return name.strip().lower()


def build_vocabulary(names: list[str]):
    characters = sorted(set("".join(names)))
    tokens = [START, END, *characters]
    token_to_id = {token: index for index, token in enumerate(tokens)}
    id_to_token = {index: token for token, index in token_to_id.items()}
    return token_to_id, id_to_token


def build_examples(names: list[str], token_to_id: dict[str, int]):
    """Create one supervised example for every character and END token."""
    contexts: list[list[int]] = []
    targets: list[int] = []
    readable_examples: list[tuple[str, str]] = []

    start_id = token_to_id[START]
    end_id = token_to_id[END]

    for name in names:
        context = [start_id] * CONTEXT_SIZE
        target_ids = [token_to_id[character] for character in name] + [end_id]

        for target_id in target_ids:
            contexts.append(context.copy())
            targets.append(target_id)

            readable_context = "".join(
                "·" if token_id == start_id else next(
                    token for token, index in token_to_id.items() if index == token_id
                )
                for token_id in context
            )
            readable_target = next(
                token for token, index in token_to_id.items() if index == target_id
            )
            readable_examples.append((readable_context, readable_target))
            context = context[1:] + [target_id]

    return (
        np.asarray(contexts, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        readable_examples,
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


@dataclass
class ForwardPass:
    embedded_flat: np.ndarray
    hidden: np.ndarray
    probabilities: np.ndarray


class TinyNameMLP:
    """Embedding -> Linear -> tanh -> Linear -> next-character logits."""

    def __init__(self, vocabulary_size: int, seed: int):
        self.vocabulary_size = vocabulary_size
        self.rng = np.random.default_rng(seed)

        input_size = CONTEXT_SIZE * EMBEDDING_SIZE
        self.parameters = {
            "embedding": self.rng.normal(
                0.0, 0.10, size=(vocabulary_size, EMBEDDING_SIZE)
            ),
            "hidden_weight": self.rng.normal(
                0.0,
                np.sqrt(2.0 / input_size),
                size=(input_size, HIDDEN_SIZE),
            ),
            "hidden_bias": np.zeros(HIDDEN_SIZE),
            "output_weight": self.rng.normal(
                0.0, 0.02, size=(HIDDEN_SIZE, vocabulary_size)
            ),
            "output_bias": np.zeros(vocabulary_size),
        }

        # Adam keeps exponentially weighted averages of each gradient and its
        # square. This usually trains more reliably than plain gradient descent.
        self.adam_first = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        self.adam_second = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        self.adam_step = 0

    @property
    def parameter_count(self) -> int:
        return sum(value.size for value in self.parameters.values())

    def forward(self, contexts: np.ndarray):
        embedding = self.parameters["embedding"]
        hidden_weight = self.parameters["hidden_weight"]
        hidden_bias = self.parameters["hidden_bias"]
        output_weight = self.parameters["output_weight"]
        output_bias = self.parameters["output_bias"]

        # (batch, context, embedding) -> (batch, context * embedding)
        embedded = embedding[contexts]
        embedded_flat = embedded.reshape(len(contexts), -1)

        # This is the hidden neural-network layer. tanh is the nonlinearity.
        hidden_pre_activation = embedded_flat @ hidden_weight + hidden_bias
        hidden = np.tanh(hidden_pre_activation)

        # One output score (logit) for every possible next token.
        logits = hidden @ output_weight + output_bias
        probabilities = softmax(logits)

        return logits, ForwardPass(embedded_flat, hidden, probabilities)

    def loss_and_gradients(self, contexts: np.ndarray, targets: np.ndarray):
        _, cache = self.forward(contexts)
        sample_count = len(contexts)

        correct_probabilities = cache.probabilities[
            np.arange(sample_count), targets
        ]
        loss = -np.mean(np.log(correct_probabilities + 1e-12))

        # Cross-entropy + softmax has a pleasantly simple derivative.
        logits_gradient = cache.probabilities.copy()
        logits_gradient[np.arange(sample_count), targets] -= 1.0
        logits_gradient /= sample_count

        output_weight = self.parameters["output_weight"]
        hidden_weight = self.parameters["hidden_weight"]

        gradients: dict[str, np.ndarray] = {}
        gradients["output_weight"] = cache.hidden.T @ logits_gradient
        gradients["output_bias"] = np.sum(logits_gradient, axis=0)

        hidden_gradient = logits_gradient @ output_weight.T
        # Derivative of tanh(x) is 1 - tanh(x)^2.
        hidden_pre_activation_gradient = hidden_gradient * (1.0 - cache.hidden**2)

        gradients["hidden_weight"] = (
            cache.embedded_flat.T @ hidden_pre_activation_gradient
        )
        gradients["hidden_bias"] = np.sum(
            hidden_pre_activation_gradient, axis=0
        )

        embedded_flat_gradient = hidden_pre_activation_gradient @ hidden_weight.T
        embedded_gradient = embedded_flat_gradient.reshape(
            sample_count, CONTEXT_SIZE, EMBEDDING_SIZE
        )

        # A character embedding can occur many times, so its gradient is the
        # sum of every position where that character appeared.
        gradients["embedding"] = np.zeros_like(self.parameters["embedding"])
        np.add.at(gradients["embedding"], contexts, embedded_gradient)

        return float(loss), gradients

    def adam_update(
        self,
        gradients: dict[str, np.ndarray],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.adam_step += 1

        for name, parameter in self.parameters.items():
            gradient = gradients[name]
            self.adam_first[name] = (
                beta1 * self.adam_first[name] + (1.0 - beta1) * gradient
            )
            self.adam_second[name] = (
                beta2 * self.adam_second[name] + (1.0 - beta2) * gradient**2
            )

            first_corrected = self.adam_first[name] / (
                1.0 - beta1**self.adam_step
            )
            second_corrected = self.adam_second[name] / (
                1.0 - beta2**self.adam_step
            )

            parameter -= learning_rate * first_corrected / (
                np.sqrt(second_corrected) + epsilon
            )

    def sample_name(
        self,
        token_to_id: dict[str, int],
        id_to_token: dict[int, str],
        temperature: float,
    ) -> str:
        start_id = token_to_id[START]
        end_id = token_to_id[END]
        context = [start_id] * CONTEXT_SIZE
        generated: list[str] = []

        for _ in range(MAX_NAME_LENGTH):
            logits, _ = self.forward(np.asarray([context], dtype=np.int64))
            next_logits = logits[0] / temperature

            # START is input padding, never a legitimate generated character.
            next_logits[start_id] = -1e9
            if len(generated) < MIN_NAME_LENGTH:
                next_logits[end_id] = -1e9

            probabilities = softmax(next_logits)
            next_id = int(self.rng.choice(self.vocabulary_size, p=probabilities))

            if next_id == end_id:
                break

            generated.append(id_to_token[next_id])
            context = context[1:] + [next_id]

        result = "".join(generated)
        return result[:1].upper() + result[1:] if result else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    names = [normalize_name(name) for name in RAW_NAMES]
    token_to_id, id_to_token = build_vocabulary(names)
    contexts, targets, readable_examples = build_examples(names, token_to_id)

    print("\nTEACHING CORPUS")
    print("---------------")
    print(", ".join(RAW_NAMES))
    print(f"Names: {len(names)}")
    print(f"Vocabulary tokens: {len(token_to_id)}")
    print(f"Next-character examples: {len(contexts)}")

    print("\nFIRST TRAINING EXAMPLES")
    print("-----------------------")
    for context, target in readable_examples[:14]:
        printable_target = "<END>" if target == END else target
        print(f"{context!r:14} -> {printable_target!r}")

    model = TinyNameMLP(vocabulary_size=len(token_to_id), seed=args.seed)
    print("\nMODEL")
    print("-----")
    print(
        f"{CONTEXT_SIZE} chars -> {EMBEDDING_SIZE}D embeddings -> "
        f"{HIDDEN_SIZE} hidden values -> {len(token_to_id)} token scores"
    )
    print(f"Learned parameters: {model.parameter_count:,}")

    print("\nTRAINING")
    print("--------")
    checkpoint_every = max(1, args.steps // 8)
    for step in range(1, args.steps + 1):
        loss, gradients = model.loss_and_gradients(contexts, targets)

        # A slightly smaller learning rate near the end helps the model settle.
        progress = step / args.steps
        learning_rate = 0.02 if progress < 0.65 else 0.006
        model.adam_update(gradients, learning_rate)

        if step == 1 or step % checkpoint_every == 0 or step == args.steps:
            print(f"step {step:4d} | loss {loss:.4f}")

    training_names = set(names)
    print("\nGENERATED NAMES")
    print("---------------")
    for temperature in (0.70, 0.95, 1.20):
        generated: list[str] = []
        attempts = 0
        while len(generated) < args.count and attempts < args.count * 100:
            attempts += 1
            candidate = model.sample_name(
                token_to_id, id_to_token, temperature=temperature
            )
            normalized_candidate = normalize_name(candidate)
            if normalized_candidate and candidate not in generated:
                generated.append(candidate)

        annotated = [
            f"{name}*" if normalize_name(name) in training_names else name
            for name in generated
        ]
        print(f"temperature {temperature:.2f}: {', '.join(annotated)}")

    print("\n* exact copy of a teaching-corpus name")
    print("Notice how lower temperatures favor memorized, probable spellings,")
    print("while higher temperatures produce more novel—and stranger—mashups.")
    print("This tiny corpus is intentionally easy to memorize.")
    print("The production version will use train/validation/test splits and")
    print("the 6,094-name deduplicated YC training file.")


if __name__ == "__main__":
    main()
