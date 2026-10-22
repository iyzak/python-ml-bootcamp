
import csv
import unittest
from pathlib import Path

import semantic_search


class TestSemanticSearch(unittest.TestCase):

    def test_search_returns_three_results(self):
        """Search should return three results for the default top_k."""

        results = semantic_search.search(
            "Which framework helps train neural networks?"
        )

        self.assertEqual(len(results), 3)

    def test_search_returns_expected_fields(self):
        """Each result should contain the required fields."""

        results = semantic_search.search(
            "How do I compare text using numerical vectors?"
        )

        required_fields = {"id", "title", "text", "score"}

        for result in results:
            self.assertTrue(required_fields.issubset(result.keys()))

    def test_search_results_are_ranked(self):
        """Results should be sorted from highest to lowest similarity."""

        results = semantic_search.search(
            "Which framework helps train neural networks?"
        )

        scores = [result["score"] for result in results]

        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_returns_pytorch_first(self):
        """The neural-network query should retrieve the PyTorch document first."""

        results = semantic_search.search(
            "Which framework helps train neural networks?"
        )

        self.assertEqual(results[0]["id"], "DOC005")

    def test_evaluation_creates_csv(self):
        """Evaluation should export its results to CSV."""

        csv_path = Path("evaluation_results.csv")

        semantic_search.run_evaluation()

        self.assertTrue(csv_path.exists())

        with csv_path.open(
            "r", newline="", encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        self.assertEqual(len(rows), 5)

        expected_columns = {
            "query",
            "retrieved_ids",
            "precision_at_1",
            "precision_at_3",
            "recall_at_3",
            "reciprocal_rank"
        }

        self.assertTrue(
            expected_columns.issubset(set(reader.fieldnames))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)