from __future__ import annotations

import unittest

from data.prepare_idea_corpus import canonical_idea, normalize_text, prepare_records


class PrepareIdeaCorpusTests(unittest.TestCase):
    def test_normalize_text_removes_urls_and_normalizes_spacing(self) -> None:
        self.assertEqual(
            normalize_text("  Tools\u00a0for teams — https://example.com  "),
            "Tools for teams",
        )

    def test_prepare_records_deduplicates_and_maps_small_categories(self) -> None:
        rows, stats = prepare_records(
            [
                {
                    "name": "Alpha",
                    "one_liner": "Scheduling for city teams.",
                    "industry": "Government",
                    "subindustry": "Government -> Operations",
                    "tags": ["Scheduling", "GovTech"],
                },
                {
                    "name": "Beta",
                    "one_liner": "Scheduling for city teams!",
                    "industry": "B2B",
                },
                {"name": "Missing", "one_liner": "", "industry": "B2B"},
                {"name": "Truncated", "one_liner": "Software for...", "industry": "B2B"},
                {"name": "Historical", "one_liner": "Acquired by Alpha.", "industry": "B2B"},
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["industry"], "Other")
        self.assertEqual(rows[0]["tags"], "Scheduling|GovTech")
        self.assertEqual(stats["duplicate_ideas_removed"], 1)
        self.assertEqual(stats["rows_dropped_missing_company_or_idea"], 1)
        self.assertEqual(stats["rows_dropped_truncated_idea"], 1)
        self.assertEqual(stats["rows_dropped_historical_status"], 1)

    def test_canonical_idea_ignores_punctuation_and_case(self) -> None:
        self.assertEqual(canonical_idea("APIs, for Teams!"), "apis for teams")


if __name__ == "__main__":
    unittest.main()
