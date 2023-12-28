from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import torch

# Define your text
text = "This is a sample sentence for classification."

# Choose a pre-trained transformer model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenize and encode the text
tokens = tokenizer(text, return_tensors="pt")

# Get the model embeddings for the input text
with torch.no_grad():
    outputs = model(**tokens)

# Extract the embeddings, you might want to process them further based on your needs
embeddings = outputs.last_hidden_state

# Now you can pass the embeddings to your downstream task or further processing

# Alternatively, you can use a transformer pipeline for text classification
classification_pipeline = pipeline(task="text-classification", model=model_name)

# Pass the text to the pipeline for classification
result = classification_pipeline(text)

# Print the result
print(result)
