
import csv
from pathlib import Path

import semantic_search


THRESHOLDS = [0.30, 0.40, 0.50, 0.60]

TEST_CASES = [
    {
        "query": "How can I identify positive and negative reviews?",
        "relevant_ids": {"DOC001", "DOC002", "DOC003"},
    },
    {
        "query": "How do I compare text using numerical vectors?",
        "relevant_ids": {"DOC004"},
    },
    {
        "query": "Which framework helps train neural networks?",
        "relevant_ids": {"DOC005"},
    },
    {
        "query": "What tool supports numerical calculations and arrays?",
        "relevant_ids": {"DOC006"},
    },
    {
        "query": "What foods make up a healthy diet?",
        "relevant_ids": {"DOC007"},
    },
    {
        "query": "How can I determine customer sentiment from feedback?",
        "relevant_ids": {"DOC001"},
    },
    {
        "query": "Which method represents sentences as vectors for comparison?",
        "relevant_ids": {"DOC004"},
    },
    {
        "query": "What library can I use to build neural networks?",
        "relevant_ids": {"DOC005"},
    },
    {
        "query": "How do I perform array calculations in Python?",
        "relevant_ids": {"DOC006"},
    },
    {
        "query": "Which foods are part of a balanced diet?",
        "relevant_ids": {"DOC007"},
    },
]


def main():
    print("Loading embedding model...")
    model = semantic_search.load_model()

    document_embeddings = (
        semantic_search.prepare_document_embeddings(
            model,
            semantic_search.documents,
        )
    )

    output_path = Path(__file__).parent / "threshold_results.csv"
    rows = []

    print("\n--- Similarity Threshold Experiment ---")

    for threshold in THRESHOLDS:
        total_returned = 0
        total_relevant_found = 0
        total_relevant_expected = 0
        queries_with_no_results = 0

        for case in TEST_CASES:
            query = case["query"]
            relevant_ids = case["relevant_ids"]

            results = semantic_search.search(
                query,
                model,
                document_embeddings,
                top_k=3,
                min_similarity=threshold,
            )

            retrieved_ids = [result["id"] for result in results]
            relevant_found = sum(
                doc_id in relevant_ids for doc_id in retrieved_ids
            )

            total_returned += len(results)
            total_relevant_found += relevant_found
            total_relevant_expected += len(relevant_ids)

            if not results:
                queries_with_no_results += 1

            rows.append({
                "threshold": threshold,
                "query": query,
                "retrieved_ids": ", ".join(retrieved_ids),
                "returned_count": len(results),
                "relevant_found": relevant_found,
                "relevant_expected": len(relevant_ids),
            })

        query_count = len(TEST_CASES)

        mean_returned = total_returned / query_count
        recall = (
            total_relevant_found / total_relevant_expected
            if total_relevant_expected else 0.0
        )

        print(f"\nThreshold: {threshold:.2f}")
        print(f"Average documents returned: {mean_returned:.2f}")
        print(f"Overall recall: {recall:.4f}")
        print(f"Queries with no results: {queries_with_no_results}")

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        fieldnames = [
            "threshold",
            "query",
            "retrieved_ids",
            "returned_count",
            "relevant_found",
            "relevant_expected",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()