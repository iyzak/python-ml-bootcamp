
import csv
from pathlib import Path

from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CSV_PATH = Path(__file__).parent / "evaluation_results.csv"

documents = [
    {
        "id": "DOC001",
        "title": "Sentiment Analysis",
        "text": "Sentiment analysis identifies customer sentiment in reviews and feedback by classifying opinions as positive, negative, or neutral. It helps determine whether customers like or dislike a product or service."
    },
    {
        "id": "DOC002",
        "title": "Hugging Face Transformers",
        "text": "Hugging Face Transformers provides pretrained models for text classification and sentiment analysis."
    },
    {
        "id": "DOC003",
        "title": "Review Classification",
        "text": "Customer review classification organizes feedback into categories such as complaints, product defects, delivery problems, and feature requests. It helps identify recurring product issues and group similar customer concerns."
    },
    {
        "id": "DOC004",
        "title": "Sentence Embeddings",
        "text": "Sentence embeddings represent text as numerical vectors that can be compared using cosine similarity."
    },
    {
        "id": "DOC005",
        "title": "PyTorch",
        "text": "PyTorch is a machine learning framework used to build and train neural networks."
    },
    {
        "id": "DOC006",
        "title": "NumPy",
        "text": "NumPy supports numerical calculations, multidimensional arrays, and mathematical operations."
    },
    {
        "id": "DOC007",
        "title": "Healthy Eating",
        "text": "A healthy diet includes vegetables, fruits, and whole grains."
    }
]


def load_model(model_name=MODEL_NAME):
    """Load and return the sentence embedding model."""
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def prepare_document_embeddings(model, document_collection):
    """Encode documents once and return normalized embeddings."""
    texts = [doc["text"] for doc in document_collection]

    return model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )


def search(
    query,
    model,
    document_embeddings,
    document_collection=documents,
    top_k=3,
    min_similarity=None
):
    """Return documents ranked by cosine similarity to a query."""

    if not query or not query.strip():
        return []

    if top_k <= 0:
        return []

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    similarities = util.cos_sim(
        query_embedding,
        document_embeddings
    )[0]

    ranked_indices = similarities.argsort(descending=True)

    results = []

    for index in ranked_indices:
        i = index.item()
        score = similarities[i].item()

        # Skip results below the optional similarity threshold.
        if min_similarity is not None and score < min_similarity:
            continue

        results.append({
            "id": document_collection[i]["id"],
            "title": document_collection[i]["title"],
            "text": document_collection[i]["text"],
            "score": score
        })

        if len(results) >= top_k:
            break

    return results

def run_evaluation(model, document_embeddings, output_path=CSV_PATH):
    """Evaluate retrieval and save per-query metrics to CSV."""

    test_cases = [
        {
            "query": "How can I identify positive and negative reviews?",
            "relevant_ids": {"DOC001", "DOC002", "DOC003"}
        },
        {
            "query": "How do I compare text using numerical vectors?",
            "relevant_ids": {"DOC004"}
        },
        {
            "query": "Which framework helps train neural networks?",
            "relevant_ids": {"DOC005"}
        },
        {
            "query": "What tool supports numerical calculations and arrays?",
            "relevant_ids": {"DOC006"}
        },
        {
            "query": "What foods make up a healthy diet?",
            "relevant_ids": {"DOC007"}
        },
        {
            "query": "How can I determine customer sentiment from feedback?",
            "relevant_ids": {"DOC001"}
        },
        {
            "query": "Which method represents sentences as vectors for comparison?",
            "relevant_ids": {"DOC004"}
        },
        {
            "query": "What library can I use to build neural networks?",
            "relevant_ids": {"DOC005"}
        },
        {
            "query": "How do I perform array calculations in Python?",
            "relevant_ids": {"DOC006"}
        },
        {
            "query": "Which foods are part of a balanced diet?",
            "relevant_ids": {"DOC007"}
        }
    ]

    metric_scores = {
        "precision_at_1": [],
        "precision_at_3": [],
        "recall_at_3": [],
        "reciprocal_rank": []
    }

    evaluation_rows = []

    print("\n--- Retrieval Evaluation ---")

    for test_case in test_cases:
        query = test_case["query"]
        relevant_ids = set(test_case["relevant_ids"])

        results = search(
            query,
            model,
            document_embeddings,
            top_k=3
        )

        retrieved_ids = [result["id"] for result in results]

        # Precision@1
        precision_at_1 = (
            1.0 if retrieved_ids and retrieved_ids[0] in relevant_ids
            else 0.0
        )

        # Precision@3: relevant retrieved documents / 3
        relevant_retrieved = sum(
            doc_id in relevant_ids for doc_id in retrieved_ids
        )
        precision_at_3 = relevant_retrieved / 3

        # Recall@3: relevant retrieved / all relevant documents
        recall_at_3 = (
            relevant_retrieved / len(relevant_ids)
            if relevant_ids else 0.0
        )

        # Reciprocal Rank: 1 / position of first relevant result
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break

        metric_scores["precision_at_1"].append(precision_at_1)
        metric_scores["precision_at_3"].append(precision_at_3)
        metric_scores["recall_at_3"].append(recall_at_3)
        metric_scores["reciprocal_rank"].append(reciprocal_rank)

        row = {
            "query": query,
            "retrieved_ids": ", ".join(retrieved_ids),
            "precision_at_1": round(precision_at_1, 4),
            "precision_at_3": round(precision_at_3, 4),
            "recall_at_3": round(recall_at_3, 4),
            "reciprocal_rank": round(reciprocal_rank, 4)
        }

        evaluation_rows.append(row)

        print(f"\nQuery: {query}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Precision@1: {precision_at_1:.4f}")
        print(f"Precision@3: {precision_at_3:.4f}")
        print(f"Recall@3: {recall_at_3:.4f}")
        print(f"Reciprocal Rank: {reciprocal_rank:.4f}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        fieldnames = [
            "query",
            "retrieved_ids",
            "precision_at_1",
            "precision_at_3",
            "recall_at_3",
            "reciprocal_rank"
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation_rows)

    print("\n--- Average Metrics ---")
    for metric, scores in metric_scores.items():
        average = sum(scores) / len(scores) if scores else 0.0
        print(f"Mean {metric}: {average:.4f}")

    print(f"\nEvaluation results saved to: {output_path}")

    return evaluation_rows

def main():
    model = load_model()
    document_embeddings = prepare_document_embeddings(model, documents)

    print("\nCustomer Review Semantic Search")
    print("Type 'eval' to evaluate retrieval.")
    print("Type 'quit' to exit.")

    while True:
        query = input("\nEnter a search query: ").strip()

        if query.lower() == "quit":
            print("Goodbye!")
            break

        if query.lower() == "eval":
            run_evaluation(model, document_embeddings)
            continue

        if not query:
            print("Please enter a search query.")
            continue

        results = search(
    query,
    model,
    document_embeddings,
    min_similarity=0.40
)

        print("\nMost relevant documents:\n")

        for result in results:
            print(f"ID: {result['id']}")
            print(f"Title: {result['title']}")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Text: {result['text']}")
            print()


if __name__ == "__main__":
    main()