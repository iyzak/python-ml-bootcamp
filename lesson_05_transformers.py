from transformers import pipeline

# Load a pretrained sentiment analysis model
classifier = pipeline("sentiment-analysis")

# Analyze a sentence
text = "This AI bootcamp is hard!"

result = classifier(text)

# Display the result
print("Text:", text)
print("Sentiment:", result[0]["label"])
print("Confidence:", round(result[0]["score"], 4))
