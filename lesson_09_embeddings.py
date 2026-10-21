
from sentence_transformers import SentenceTransformer

# Load a small pretrained embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "I want to purchase a laptop.",
    "I need to buy a computer.",
    "The weather is sunny today."
]

# Convert sentences into numerical vectors
embeddings = model.encode(sentences)

print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)

# Compare the first two sentence embeddings
similarity = model.similarity(
    embeddings[0:1],
    embeddings[1:2]
)

print("Similarity between sentence 1 and 2:")
print(round(similarity.item(), 4))
