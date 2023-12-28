from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, pipeline
import torch

# Initialize GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", config=GPT2Config(use_return_dict=True))

# Define a sequence
sequence = "Hello, how are you today?"

# Tokenize the sequence
tokenized_sequence = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)

# Obtain token embeddings from the GPT-2 model
with torch.no_grad():
    model_output = model(**tokenized_sequence)

# Extract the last layer hidden states (you may customize this based on your specific needs)
last_hidden_states = model_output.last_hidden_state

# Generate positional encoding (this is a simplified example)
positional_encoding = torch.randn_like(last_hidden_states)

# Add positional encoding to token embeddings
combined_embeddings = last_hidden_states + positional_encoding

# Use the pipeline for text generation (this is just an example, and you can adapt it for your specific task)
text_generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
generated_text = text_generation_pipeline("The combined embeddings are: " + tokenizer.decode(combined_embeddings[0]))

# Print the generated text
print(generated_text[0]['generated_text'])
