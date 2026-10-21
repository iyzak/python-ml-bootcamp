
from transformers import pipeline

# Load a pretrained sentiment model
classifier = pipeline("sentiment-analysis")

# Sample customer reviews
reviews = [
    "The application is easy to use.",
    "The application keeps crashing.",
    "I expected better performance.",
    "The new update is excellent!",
    "Customer support never responds."
]

# Analyze every review
for review in reviews:
    result = classifier(review)[0]

    sentiment = result["label"]
    score = result["score"]

    print("\nReview:", review)
    print("Sentiment:", sentiment)
    print("Score:", round(score, 4))
