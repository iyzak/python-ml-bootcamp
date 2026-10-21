
from transformers import pipeline

print("Loading the sentiment analysis model...")
classifier = pipeline("sentiment-analysis")
print("Model ready!")

review_count = 0

while True:
    text = input("\nEnter a sentence (or type 'quit' to exit): ")

    if text.lower().strip() == "quit":
        print(f"\nYou analyzed {review_count} review(s) in total.")
        print("Goodbye!")
        break

    if not text.strip():
        print("Please enter some text.")
        continue

    result = classifier(text)[0]
    review_count += 1

    label = result["label"]
    score = result["score"]

    print("Sentiment:", label)
    print("Score:", round(score, 4))