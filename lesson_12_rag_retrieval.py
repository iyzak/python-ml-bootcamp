
from sentence_transformers import SentenceTransformer, util

print("Loading the embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Knowledge base: short documents about AI and Python
documents = [
    {
        "title": "PyTorch",
        "text": "PyTorch is a machine learning framework used to build and train deep learning and neural network models."
    },
    {
        "title": "NumPy",
        "text": "NumPy supports numerical computing in Python, including arrays, mathematical operations, and statistical calculations."
    },
    {
        "title": "Embeddings",
        "text": "Embeddings represent text as numerical vectors that capture aspects of meaning. They help systems compare and retrieve relevant text."
    },
    {
        "title": "Transformers",
        "text": "Hugging Face Transformers provides tools and pretrained models for tasks such as text classification, language understanding, and text generation."
    },
    {
        "title": "Semantic Search",
        "text": "Semantic search retrieves information by comparing the meaning of a query with the meaning of stored documents."
    }
]

# Convert document text into embeddings
document_texts = [doc["text"] for doc in documents]

document_embeddings = model.encode(
    document_texts,
    convert_to_tensor=True
)

print("Knowledge base ready!")

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

    # Convert the question into an embedding
    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    # Compare the question with all documents
    scores = util.cos_sim(
        query_embedding,
        document_embeddings
    )[0]

    # Retrieve the two most relevant documents
    top_results = scores.argsort(
        descending=True
    )[:2]

    print("\nRetrieved context:\n")

    for index in top_results:
        i = index.item()

        print("Document:", documents[i]["title"])
        print("Similarity:", round(scores[i].item(), 4))
        print("Text:", documents[i]["text"])
        print()

    print("These passages are ready to be passed to an AI model.")
