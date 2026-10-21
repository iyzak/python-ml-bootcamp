
from transformers import pipeline

# Load a pretrained sentiment-analysis model
classifier = pipeline("sentiment-analysis")

# Test several sentences
sentences = [
    "The application is easy to use.",
    "The application keeps crashing.",
    "I expected better performance."
]

for sentence in sentences:
    result = classifier(sentence)[0]

    print("\nText:", sentence)
    print("Sentiment:", result["label"])
    print("Score:", round(result["score"], 4))
