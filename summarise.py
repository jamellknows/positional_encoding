from transformers import BartForConditionalGeneration, BartTokenizer

def summarize_text(inputs, max_length=150, model_name="facebook/bart-large-cnn"):
    # Load pre-trained BART model and tokenizer
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Tokenize and encode the text
    

    # Generate the summary
    summary_ids = model.generate(**inputs, max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Example of preprocessed text
# preprocessed_text = "Your preprocessed text here. This could be the output of any preprocessing steps."

# Perform text summarization
# generated_summary = summarize_text(preprocessed_text)

# Print the results
# print("Original Text:")
# print(preprocessed_text)
# print("\nGenerated Summary:")
# print(generated_summary)
