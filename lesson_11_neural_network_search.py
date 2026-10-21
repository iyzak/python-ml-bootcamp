
from sentence_transformers import SentenceTransformer, util

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Our small document collection
documents = [
    "Python is a programming language used to build AI applications.",
    "PyTorch helps developers train deep learning models.",
    "NumPy is useful for numerical calculations and arrays.",
    "Hugging Face provides tools for working with language models.",
    "A healthy diet includes vegetables, fruits, and whole grains."
]

# The user's search query
query = "Which tool can I use to train a neural network?"

# Convert documents and query into embeddings
document_embeddings = model.encode(documents, convert_to_tensor=True)
query_embedding = model.encode(query, convert_to_tensor=True)

# Compare the query with every document
similarities = util.cos_sim(query_embedding, document_embeddings)[0]

# Rank documents from most to least relevant
ranked_results = similarities.argsort(descending=True)

print("\nSearch query:", query)
print("\nMost relevant documents:\n")

for index in ranked_results:
    i = index.item()
    score = similarities[i].item()

    print(f"Similarity: {score:.4f}")
    print(f"Document: {documents[i]}")
    print()
