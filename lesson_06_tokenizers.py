
from transformers import AutoTokenizer

# Use the tokenizer associated with a pretrained model
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "AI is transforming healthcare."

# Convert text into token pieces
tokens = tokenizer.tokenize(text)

# Convert those pieces into numerical IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Encode the text in the format expected by the model
encoded = tokenizer(text)

print("Original text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Input IDs:", encoded["input_ids"])
print("Decoded text:", tokenizer.decode(encoded["input_ids"]))

new_text = "AI is useful."

new_encoded = tokenizer(new_text)

print("\nNew text:", new_text)
print("New tokens:", tokenizer.tokenize(new_text))
print("New input IDs:", new_encoded["input_ids"])
