from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
import numpy as np 

def sentiment(text):
    blob = TextBlob(text)
    analyzer = SentimentIntensityAnalyzer()

    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    if sentiment_polarity == 0 or sentiment_subjectivity == 0:
        sentiment_scores = analyzer.polarity_scores(text)
        non_zero_indices = [index for index, value in sentiment_scores.items() if value != 0.0]
        max_key = max(sentiment_scores, key=lambda k: sentiment_scores[k])
        max_value = max(sentiment_scores.values())
        if max_value == 0.0:
            max_value = 1.0
        if max_key == 'neg':
            sentiment_polarity = -max_value
        else:
            sentiment_polarity = max_value 
    
    arr = np.array([text, sentiment_polarity, sentiment_subjectivity, 0, 0])

    
    return arr

# Example usage

