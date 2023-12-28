# import tokensisers
import numpy as np
from split_word import split_word
from transformers import AutoTokenizer
# Specify the model name or path
model_name = "bert-base-uncased"

# Use AutoTokenizer to load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize a sentence
# sentence = "Hello, how are you?"
# tokens = tokenizer.tokenize(sentence)
# my_tokens = split_word(sentence)

# print("Original Sentence:", sentence)
# print("Tokenized Result:", tokens)
# print("My tokeniser: ", my_tokens)


# datasets
from datasets import load_dataset
import csv


def read_csv(file_path):
    data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            data.append(row)
    return data

summarisation_dataset = load_dataset("TanveerAman/AMI-Corpus-Text-Summarization")
summarisation_dataset = summarisation_dataset['train']['Dialogue'][:30]
sentiment_dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
sentiment_dataset= sentiment_dataset['train']['text'][:30]

# generation_dataset_prompts = []
# prompts = 
# for entry in generation_dataset:
#     prompts = entry['prompts']
#     generation_dataset_prompts.append(prompts)

# datasets = [summarisation_dataset, sentiment_dataset, generation_dataset]
# tokenise dataset 
from nltk.tokenize import word_tokenize
import nltk
import torch
nltk.download('punkt')
tokenized_summaraisation = [word_tokenize(text) for text in summarisation_dataset]
tokenized_sentiment = [word_tokenize(text) for text in sentiment_dataset]
tokenized = [tokenized_summaraisation, tokenized_sentiment]

# import embedders 

from sentimental_embedding import embedding_sentence_per_word
from gensim.models import Word2Vec
sen_embedded_summarisation = embedding_sentence_per_word(summarisation_dataset)
sen_embedded_sentiment = embedding_sentence_per_word(sentiment_dataset)

models = []
for dataset in tokenized: 
    model = Word2Vec(dataset, vector_size=20, window=5, min_count=1, workers=4)
    models.append(model)
# models [0] semantic embedded summaraistion models[1] is semantic embedded sentimental
# pass through embedder 
# sen_embedding = embedding_sentence_per_word()
# # print("Model WV", model.wv.vectors)
# text_embeddings_conventional = model.wv.vectors
# text_embeddings_my = embedding_sentence_per_word(my_tokens)

# print("Mine", text_embeddings_my)

# positional encodings 

from sinusoidal_encoding import getSinusoidalPositionEncoding
from henon_map_encoding import getHenonMapEncoding
labels =["semantic-embedded summarisation", "semantic-embedded sentiment", "sentimental-embedded summarisation", "sentimental-embedded sentiment"]

#  for each dataset embedding 
# sinusodial and the henon 
# 4 embeddings and herefore 8 encodings 
#  embedding + encoding element wise 
# summarisation model 
# sentimenet model

sem_emb_sum_sin_pos_enc = getSinusoidalPositionEncoding(len(models[0].wv.vectors),4)
sem_emb_sen_sin_pos_enc = getSinusoidalPositionEncoding(len(models[1].wv.vectors),4)
sen_emb_sum_sin_pos_enc = getSinusoidalPositionEncoding(len(sen_embedded_summarisation),4)
sen_emb_sen_sin_pos_enc = getSinusoidalPositionEncoding(len(sen_embedded_sentiment),4)

sem_emb_sum_hen_pos_enc = getHenonMapEncoding(models[0].wv.vectors)
sem_emb_sen_hen_pos_enc = getHenonMapEncoding(models[1].wv.vectors)
sen_emb_sum_hen_pos_enc = getHenonMapEncoding(sen_embedded_summarisation)
sen_emb_sen_hen_pos_enc = getHenonMapEncoding(sen_embedded_sentiment)

from summarise import summarize_text
# print(type(sem_emb_sum_sin_pos_enc))
# tile arrays to fit embedding 
sem_emb_sum_sin_pos_enc_tiled = np.tile(sem_emb_sum_sin_pos_enc, (1,5))
sem_emb_sen_sin_pos_enc_tiled = np.tile(sem_emb_sen_sin_pos_enc, (1,5))
# sem_emb_sum_hen_pos_enc_tiled = np.tile(sem_emb_sum_hen_pos_enc, (1,5))
# sem_emb_sen_hen_pos_enc_tiled = np.tile(sem_emb_sen_hen_pos_enc, (1,5))

sem_emb_sum_sin_enc = sem_emb_sum_sin_pos_enc_tiled + models[0].wv.vectors
sem_emb_sen_sin_enc = sem_emb_sen_sin_pos_enc_tiled + models[1].wv.vectors

# print(sem_emb_sum_sin_enc)

# print(type(sen_emb_sum_sin_pos_enc))
# print(type(sen_embedded_summarisation))
# print(sen_embedded_summarisation)
sen_emb_sum_sin_pos_enc_two_n_plus_one = np.hstack([sen_emb_sum_sin_pos_enc,sen_emb_sum_sin_pos_enc[:,:2]])
sen_emb_sum_sin_pos_enc_two_n_plus_one_tiled = np.tile(sen_emb_sum_sin_pos_enc_two_n_plus_one, (1,5))
sen_emb_sum_sin_enc = sen_emb_sum_sin_pos_enc_two_n_plus_one_tiled + sen_embedded_summarisation.astype(np.complex64)


sen_emb_sen_sin_pos_enc_two_n_plus_one = np.hstack([sen_emb_sen_sin_pos_enc, sen_emb_sen_sin_pos_enc[:,:2]])
sen_emb_sen_sin_pos_enc_two_n_plus_one_tiled = np.tile(sen_emb_sen_sin_pos_enc_two_n_plus_one, (1,5))
sen_emb_sen_sin_enc = sen_emb_sen_sin_pos_enc_two_n_plus_one_tiled + sen_embedded_sentiment.astype(np.complex64)



sem_emb_sum_hen_enc = sem_emb_sum_hen_pos_enc[:, :20] + models[0].wv.vectors
sem_emb_sen_hen_enc = sem_emb_sen_hen_pos_enc[:, :20] + models[1].wv.vectors

sen_emb_sum_hen_pos_enc_two_n_plus_one = np.hstack([sen_emb_sum_hen_pos_enc,sen_emb_sum_hen_pos_enc[:,:6]])
sen_emb_sum_hen_enc = sen_emb_sum_hen_pos_enc_two_n_plus_one + sen_embedded_summarisation.astype(np.complex64)

sen_emb_sen_hen_pos_enc_two_n_plus_one = np.hstack([sen_emb_sen_hen_pos_enc,sen_emb_sen_hen_pos_enc[:,:6]])

sen_emb_sen_hen_enc = sen_emb_sen_hen_pos_enc_two_n_plus_one + sen_embedded_sentiment.astype(np.complex64)

encoding_array = {"Semantic Embedded Sinusoidal Encoded Summation Data":sem_emb_sum_sin_enc, "Semantic Embedded Sinusoidal Encoded Sentimental Data": sem_emb_sen_sin_enc, "Sentimental Embedded Sinusoidal Encoded Summation Data": sen_emb_sum_sin_enc, "Sentimental Embedded Sinusoidal Encoded Sentimental Data":sen_emb_sen_sin_enc, "Semantic Embedded Henon Encoded Summation Data":sem_emb_sum_hen_enc, "Semantic Embedded Henon Encoded Sentimental Data": sem_emb_sen_hen_enc, "Sentimental Embedded Henon Encoded Summation Data":sen_emb_sum_hen_enc, "Sentimental Embedded Henon Encoded Sentimental Data": sen_emb_sen_hen_enc}

from classify import  run_tests
from KNN import KNN
from transformer import transformer_trainer

# Semantic vs Sentiment - Embedding - Sinusoidal Encoding - Summation -1
# X = sem_emb_sum_sin_enc[:4, :]
# y = sen_emb_sum_sin_enc.real[:,:20]
# validate_1 = sen_embedded_sentiment[:,:20].astype(np.complex64).real
# validate_2 = models[0].wv.vectors[:4,:]
# print("Semantic vs Sentiment - Embedding - Sinusoidal Encoding - Summation Training Sentimental Validation")
# print(f"X data is : {X} \n")
# print(f"y data is : {y} \n")
# print(f"validation data is {validate_1} \n")
# print(f"validation to compare against data is {validate_2}\n")
# rf, br = run_tests(X,y,validate_1)
# knn = KNN(X,y,validate_1)
# X = sem_emb_sum_sin_enc[:4, 0:4]
# y = sen_emb_sum_sin_enc.real[:,:4]
# fit, summary, prediction = transformer_trainer(X,y)
# data = [rf, br, knn]
# data2 = [fit, summary, prediction]
# import pandas as pd









from henonTransformer import henonTransformer
from henonPipe import henonPipe
# Semantic vs Sentiment - Embedding - Henon Encoding - sentimental - test on summation -2 - redo

# X = sem_emb_sum_hen_enc[:4, :]
# y = sen_emb_sum_hen_enc.real[:,:20]
# validate_1 = sem_emb_sen_hen_enc[:4, :]
# validate_2 = sen_emb_sen_hen_enc.real[:,:20]
# print("Semantic vs Sentiment - Embedding - Henon Encoding - Summation Training Sentimental Validation")
# print(f"X data is : {X} \n")
# print(f"y data is : {y} \n")
# print(f"validation data is {validate_1} \n")
# print(f"validation to compare against data is {validate_2}\n")
# run_tests(X,y,validate_1)
# KNN(X,y,validate_1)
# X = sem_emb_sum_hen_enc[:4, 0:4]
# y = sen_emb_sum_hen_enc.real[:,:4]
# X.astype(np.float64)
# y.astype(np.float64)
# henonTransformer(X,y, validate_1,validate_2)


# Semantic vs Sentiment - Embedding - Henon Encoding - Sentimental Training Summation Validation - 3 redo 

# X = sem_emb_sum_sin_enc.real[:4,:20]
# y = sem_emb_sum_hen_enc[:4,:20]
# validate_1 = sem_emb_sen_sin_enc.real[:4,:20]
# validate_2 = sem_emb_sen_hen_enc.real[:4,:20]
# print("Semantic vs Sentiment - Embedding - Henon Encoding - Sentimental Training Summation Validation")
# print(f"X data is : {X} \n")
# print(f"y data is : {y} \n")
# print(f"validation data is {validate_1} \n")
# print(f"validation to compare against data is {validate_2}\n")
# run_tests(X,y,validate_1)
# KNN(X,y,validate_1)
# X = sen_emb_sum_hen_enc[:, 0:4]
# y = sem_emb_sum_hen_enc.real[:4,:4]
# X.astype(np.float64)
# y.astype(np.float64)
# henonTransformer(X,y, validate_1,validate_2)

#  Sinusoidal vs Henon - Semantic Embedding - Summation Training vs Sentimental Validation - 4 redo

X = sen_emb_sen_hen_enc.real[:4,:20]
y = sen_emb_sum_sin_enc.real[:,:20]
validate_1 = sen_emb_sum_hen_enc.real[:4,:20]
validate_2 = sen_emb_sum_sin_enc.real[:,:20]
print("Semantic vs Sentiment - Embedding - Henon Encoding - Sentimental Training Summation Validation")
print(f"X data is : {X} \n")
print(f"y data is : {y} \n")
print(f"validation data is {validate_1} \n")
print(f"validation to compare against data is {validate_2}\n")
run_tests(X,y,validate_1)
KNN(X,y,validate_1)
X = sem_emb_sum_sin_enc.real[:4, 0:4]
y = sem_emb_sen_hen_enc.real[:4,:4]
X.astype(np.float64)
y.astype(np.float64)
henonTransformer(X,y, validate_1,validate_2)


#  Sinusoidal vs Henon - Semantic Embedding - Sentimental Training vs Summation Validation - 6 

# X = sem_emb_sen_sin_enc.real[:4,:]
# y = sem_emb_sen_hen_enc[:,:20]
# # validate_1 = 
# validate_2 = models[0].wv.vectors[:,:20]
# print("Semantic vs Sentiment - Embedding - Henon Encoding - Sentimental Training Summation Validation")
# print(f"X data is : {X} \n")
# print(f"y data is : {y} \n")
# print(f"validation data is {validate_2} \n")
# run_tests(X,y,validate_2)
# KNN(X,y,validate_2)
# X = sem_emb_sen_sin_enc.real[:4, :]
# y = sem_emb_sen_hen_enc[:,:20]
# X.astype(np.float64)
# y.astype(np.float64)
# # X = X.reshape(-1, 3, X.shape[1])

# henonTransformer(X,y, validate_2,validate_1)
# # henonPipe(X,y,validate_2)
# encoding_array = {"Semantic Embedded Sinusoidal Encoded Summation Data":sem_emb_sum_sin_enc, "Semantic Embedded Sinusoidal Encoded Sentimental Data": sem_emb_sen_sin_enc, "Sentimental Embedded Sinusoidal Encoded Summation Data": sen_emb_sum_sin_enc, "Sentimental Embedded Sinusoidal Encoded Sentimental Data":sen_emb_sen_sin_enc, "Semantic Embedded Henon Encoded Summation Data":sem_emb_sum_hen_enc, "Semantic Embedded Henon Encoded Sentimental Data": sem_emb_sen_hen_enc, "Sentimental Embedded Henon Encoded Summation Data":sen_emb_sum_hen_enc, "Sentimental Embedded Henon Encoded Sentimental Data": sen_emb_sen_hen_enc}

#  Sinusoidal vs Henon - Sentimental Embedding - Summation Training vs Sentimental Validation - 7

# X = sen_emb_sum_sin_enc.real[:4,:]
# y = sen_emb_sum_hen_enc[:,:20]
# validate_1 = models[0].wv.vectors[:,:20]
# validate_2 = sen_embedded_sentiment[:4,:].astype(np.complex64).real
# print("Semantic vs Sentiment - Embedding - Henon Encoding - Sentimental Training Summation Validation")
# print(f"X data is : {X} \n")
# print(f"y data is : {y} \n")
# print(f"validation data is {validate_2} \n")
# run_tests(X,y,validate_2)
# KNN(X,y,validate_2)
# X = sem_emb_sum_sin_enc.real[:, 0:4]
# y = sem_emb_sen_hen_enc[:4,:4]
# X.astype(np.float64)
# y.astype(np.float64)
# # X = X.reshape(-1, 3, X.shape[1])

# henonTransformer(X,y, validate_2,validate_1)
# henonPipe(X,y,validate_2)
#  Sinusoidal vs Henon - Sentimental Embedding - Sentimental Training vs Summation Validation - 8






