from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from model.idea_model import (
    IdeaLanguageModel,
    NoveltyIndex,
    display_idea,
    generate_ideas,
    idea_similarity,
    is_well_formed_idea,
    normalize_idea,
    tokenize,
)


SYNTHETIC_ROWS = [
    {"industry": "B2B", "idea": "Analytics software for independent dental clinics"},
    {"industry": "B2B", "idea": "Billing software for independent dental clinics"},
    {"industry": "B2B", "idea": "Analytics software for regional veterinary clinics"},
    {"industry": "B2B", "idea": "Scheduling tools for regional veterinary clinics"},
    {"industry": "Consumer", "idea": "A marketplace for local cooking classes"},
    {"industry": "Consumer", "idea": "A community for local hiking groups"},
    {"industry": "Consumer", "idea": "A marketplace for neighborhood fitness classes"},
]


class IdeaModelTests(unittest.TestCase):
    def test_tokenization_and_display_are_stable(self) -> None:
        tokens = tokenize("AI-powered tools for clinics.")
        self.assertEqual(tokens, ["ai-powered", "tools", "for", "clinics", "."])
        self.assertEqual(display_idea(tokens), "AI-powered tools for clinics.")
        self.assertEqual(normalize_idea("Tools, for clinics!"), "tools for clinics")

    def test_novelty_index_finds_lightly_edited_copy(self) -> None:
        known = "Workflow automation for independent dental clinics"
        candidate = "Workflow automation for regional dental clinics"
        index = NoveltyIndex([known])
        nearest, score = index.nearest(candidate)
        self.assertEqual(nearest, known)
        self.assertGreater(score, 0.55)
        self.assertGreater(idea_similarity(known, candidate), 0.55)

    def test_model_round_trip_preserves_seeded_sample(self) -> None:
        model = IdeaLanguageModel.train(SYNTHETIC_ROWS, min_context_count=1)
        before = model.sample(
            category="B2B",
            random_source=random.Random(7),
            creativity="medium",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json.gz"
            model.save(path)
            restored = IdeaLanguageModel.load(path)
            after = restored.sample(
                category="B2B",
                random_source=random.Random(7),
                creativity="medium",
            )
        self.assertEqual(before, after)

    def test_generation_rejects_training_copies(self) -> None:
        model = IdeaLanguageModel.train(SYNTHETIC_ROWS, min_context_count=1)
        results = generate_ideas(
            model,
            count=2,
            seed=21,
            category="B2B",
            creativity="high",
            similarity_limit=0.95,
            diversity_limit=0.9,
        )
        known = {normalize_idea(row["idea"]) for row in SYNTHETIC_ROWS}
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(normalize_idea(result.idea) not in known for result in results))
        self.assertTrue(all(is_well_formed_idea(result.idea) for result in results))

    def test_validation_rejects_repeated_phrases(self) -> None:
        self.assertFalse(is_well_formed_idea("AI agents for AI agents."))


if __name__ == "__main__":
    unittest.main()
