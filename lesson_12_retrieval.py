
from sentence_transformers import SentenceTransformer, util

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# A small knowledge base with source IDs
documents = [
    {
        "id": "DOC001",
        "title": "PyTorch",
        "text": "PyTorch is a machine learning framework used to build and train deep learning and neural network models."
    },
    {
        "id": "DOC002",
        "title": "Python",
        "text": "Python is a programming language used to build AI applications, automate tasks, and process data."
    },
    {
        "id": "DOC003",
        "title": "NumPy",
        "text": "NumPy supports numerical calculations, multidimensional arrays, and mathematical operations in Python."
    },
    {
        "id": "DOC004",
        "title": "Transformers",
        "text": "Hugging Face Transformers provides pretrained models for text classification, language understanding, and text generation."
    },
    {
        "id": "DOC005",
        "title": "Embeddings",
        "text": "Text embeddings represent text as numerical vectors that can be compared to find semantically similar content."
    },
    {
        "id": "DOC006",
        "title": "RAG",
        "text": "Retrieval-Augmented Generation retrieves relevant document passages to provide context for a language model."
    },
    {
        "id": "DOC007",
        "title": "Dairy monitoring",
        "text": "A smart dairy system can monitor animal health, feeding, and milk production using sensors and data analytics."
    }
]

# Prepare document text and calculate embeddings once
document_texts = [doc["text"] for doc in documents]

document_embeddings = model.encode(
    document_texts,
    convert_to_tensor=True
)

print(f"Knowledge base ready: {len(documents)} documents.")


while True:
    query = input(
        "\nAsk a question (or type 'quit' to exit): "
    ).strip()

    if query.lower() == "quit":
        print("Goodbye!")
        break

    if not query:
        print("Please enter a question.")
        continue

    # Embed the question
    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    # Retrieve the top 3 matching documents
    results = util.semantic_search(
        query_embedding,
        document_embeddings,
        top_k=3
    )[0]

    # Only show matches above this threshold
    threshold = 0.30

    relevant_results = [
        result for result in results
        if result["score"] >= threshold
    ]

    print("\nRetrieved context:")

    if not relevant_results:
        print("No sufficiently relevant documents found.")
        print("Try rephrasing your question.")
        continue

    for result in relevant_results:
        doc = documents[result["corpus_id"]]

        print(f"\nSource: {doc['id']}")
        print(f"Title: {doc['title']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Text: {doc['text']}")

    print("\nThese passages are ready for a future RAG answer generator.")
