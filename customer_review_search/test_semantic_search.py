
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import semantic_search


class FakeModel:
    """A small deterministic model for fast unit tests."""

    def encode(
        self,
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    ):
        if isinstance(texts, str):
            texts = [texts]

        vectors = []

        for text in texts:
            text = text.lower()

            if (
                "neural" in text
                or "pytorch" in text
                or "framework" in text
            ):
                vectors.append([1.0, 0.0, 0.0])
            elif (
                "healthy" in text
                or "food" in text
                or "diet" in text
            ):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])

        tensor = torch.tensor(vectors, dtype=torch.float32)

        if normalize_embeddings:
            tensor = torch.nn.functional.normalize(
                tensor, p=2, dim=1
            )

        return tensor


class TestSearchUnit(unittest.TestCase):

    def setUp(self):
        self.model = FakeModel()

        self.test_documents = [
            {
                "id": "TEST001",
                "title": "PyTorch",
                "text": "PyTorch is a framework for neural networks."
            },
            {
                "id": "TEST002",
                "title": "Healthy Eating",
                "text": "A healthy diet includes vegetables and fruits."
            },
            {
                "id": "TEST003",
                "title": "General Information",
                "text": "This document contains general information."
            }
        ]

        self.embeddings = semantic_search.prepare_document_embeddings(
            self.model,
            self.test_documents
        )

    def test_search_returns_three_results(self):
        results = semantic_search.search(
            "Which framework trains neural networks?",
            self.model,
            self.embeddings,
            self.test_documents
        )

        self.assertEqual(len(results), 3)

    def test_search_returns_expected_fields(self):
        results = semantic_search.search(
            "Which framework trains neural networks?",
            self.model,
            self.embeddings,
            self.test_documents
        )

        expected_fields = {"id", "title", "text", "score"}

        for result in results:
            self.assertTrue(expected_fields.issubset(result.keys()))

    def test_search_ranks_matching_document_first(self):
        results = semantic_search.search(
            "Which framework trains neural networks?",
            self.model,
            self.embeddings,
            self.test_documents
        )

        self.assertEqual(results[0]["id"], "TEST001")

    def test_search_scores_are_descending(self):
        results = semantic_search.search(
            "Which framework trains neural networks?",
            self.model,
            self.embeddings,
            self.test_documents
        )

        scores = [result["score"] for result in results]

        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query_returns_no_results(self):
        results = semantic_search.search(
            "   ",
            self.model,
            self.embeddings,
            self.test_documents
        )

        self.assertEqual(results, [])

    def test_non_positive_top_k_returns_no_results(self):
        results = semantic_search.search(
            "Which framework trains neural networks?",
            self.model,
            self.embeddings,
            self.test_documents,
            top_k=0
        )

        self.assertEqual(results, [])

    def test_top_k_limits_result_count(self):
        results = semantic_search.search(
            "Which framework trains neural networks?",
            self.model,
            self.embeddings,
            self.test_documents,
            top_k=1
        )

        self.assertEqual(len(results), 1)

    def test_similarity_threshold_keeps_relevant_results(self):
        results = semantic_search.search(
            "Which framework helps train neural networks?",
            self.model,
            self.embeddings,
            self.test_documents,
            min_similarity=0.40
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], "TEST001")
        self.assertTrue(
            all(result["score"] >= 0.40 for result in results)
        )

    def test_similarity_threshold_filters_low_scores(self):
        results = semantic_search.search(
            "Which framework helps train neural networks?",
            self.model,
            self.embeddings,
            self.test_documents,
            min_similarity=1.01
        )

        self.assertEqual(results, [])

    def test_similarity_threshold_can_return_fewer_than_top_k(self):
        results = semantic_search.search(
            "Which framework helps train neural networks?",
            self.model,
            self.embeddings,
            self.test_documents,
            top_k=3,
            min_similarity=0.40
        )

        self.assertLess(len(results), 3)
        self.assertTrue(
            all(result["score"] >= 0.40 for result in results)
        )


class TestEvaluationUnit(unittest.TestCase):

    def test_evaluation_writes_rows_to_csv(self):
        fake_result = [{
            "id": "DOC001",
            "title": "Sentiment Analysis",
            "text": "Example document",
            "score": 0.9
        }]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_results.csv"

            with patch(
                "semantic_search.search",
                return_value=fake_result
            ):
                rows = semantic_search.run_evaluation(
                    FakeModel(),
                    torch.tensor([[1.0, 0.0, 0.0]]),
                    output_path=csv_path
                )

            self.assertTrue(csv_path.exists())
            self.assertGreater(len(rows), 0)

            with csv_path.open(
                "r",
                newline="",
                encoding="utf-8"
            ) as file:
                reader = csv.DictReader(file)
                csv_rows = list(reader)
                columns = reader.fieldnames

            self.assertEqual(len(csv_rows), len(rows))

            expected_columns = {
                "query",
                "retrieved_ids",
                "precision_at_1",
                "precision_at_3",
                "recall_at_3",
                "reciprocal_rank"
            }

            self.assertTrue(
                expected_columns.issubset(set(columns))
            )


class TestRealModelIntegration(unittest.TestCase):
    """Integration tests using the actual pretrained embedding model."""

    @classmethod
    def setUpClass(cls):
        cls.model = semantic_search.load_model()
        cls.embeddings = semantic_search.prepare_document_embeddings(
            cls.model,
            semantic_search.documents
        )

    def test_real_model_retrieves_pytorch_first(self):
        results = semantic_search.search(
            "Which framework helps train neural networks?",
            self.model,
            self.embeddings,
            semantic_search.documents
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], "DOC005")

    def test_real_model_returns_descending_scores(self):
        results = semantic_search.search(
            "How do I compare text using numerical vectors?",
            self.model,
            self.embeddings,
            semantic_search.documents
        )

        scores = [result["score"] for result in results]

        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)