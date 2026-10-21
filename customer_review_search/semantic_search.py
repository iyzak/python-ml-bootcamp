
from sentence_transformers import SentenceTransformer, util

# 1. Load the sentence embedding model
print("Loading the embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Build a document collection focused on customer reviews
documents = [
    {
        "id": "DOC001",
        "title": "Sentiment Analysis",
        "text": "Sentiment analysis classifies customer reviews as positive, negative, or neutral."
    },
    {
        "id": "DOC002",
        "title": "Hugging Face Transformers",
        "text": "Hugging Face Transformers provides pretrained models for text classification and sentiment analysis."
    },
    {
        "id": "DOC003",
        "title": "Review Classification",
        "text": "Customer feedback can be analyzed to identify complaints, satisfaction, and common product issues."
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

# 3. Encode the documents once
document_texts = [doc["text"] for doc in documents]

document_embeddings = model.encode(
    document_texts,
    convert_to_tensor=True,
    normalize_embeddings=True
)


def search(query, top_k=3):
    """Return the top-k documents most similar to a query."""

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    similarities = util.cos_sim(
        query_embedding, document_embeddings
    )[0]

    top_k = min(top_k, len(documents))
    ranked_indices = similarities.argsort(descending=True)[:top_k]

    results = []

    for index in ranked_indices:
        i = index.item()

        results.append({
            "id": documents[i]["id"],
            "title": documents[i]["title"],
            "text": documents[i]["text"],
            "score": similarities[i].item()
        })

    return results


def run_evaluation():
    """Check whether known relevant documents appear in the top results."""

    test_cases = [
        {
            "query": "How can I identify positive and negative reviews?",
            "relevant_ids": {"DOC001", "DOC002", "DOC003"}
        },
        {
            "query": "How do I compare text using numerical vectors?",
            "relevant_ids": {"DOC004"}
        }
    ]

    print("\n--- Retrieval Evaluation ---")

    for test in test_cases:
        results = search(test["query"], top_k=3)
        retrieved_ids = {result["id"] for result in results}
        relevant_ids = test["relevant_ids"]

        hits = len(retrieved_ids & relevant_ids)
        recall_at_3 = hits / len(relevant_ids)

        print(f"\nQuery: {test['query']}")
        print(f"Relevant documents found: {hits}/{len(relevant_ids)}")
        print(f"Recall@3: {recall_at_3:.2f}")


def main():
    print("\nCustomer Review Semantic Search")
    print("Type 'eval' to evaluate retrieval.")
    print("Type 'quit' to exit.")

    while True:
        query = input("\nEnter a search query: ").strip()

        if query.lower() == "quit":
            print("Goodbye!")
            break

        if query.lower() == "eval":
            run_evaluation()
            continue

        if not query:
            print("Please enter a search query.")
            continue

        results = search(query)

        print("\nMost relevant documents:\n")

        for result in results:
            print(f"ID: {result['id']}")
            print(f"Title: {result['title']}")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Text: {result['text']}")
            print()


if __name__ == "__main__":
    main()
