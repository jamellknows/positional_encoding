
# split the text -> get the sentiment of each word -> create the array on the results 
import sentimental_analyser as sa
import numpy as np 
import math 
import cmath
def calculate_stats(array):
    if len(array) == 0:
        raise ValueError("Array must not be empty")

    # Convert the array to a numpy array for easy calculations
    np_array = np.array(array, dtype=complex)

    # Calculate average
    average = np.mean(np_array) / np.linalg.norm(np_array)

    # Calculate standard deviation
    std_deviation = np.std(np_array) / np.linalg.norm(np_array)

    # Calculate the difference between the largest and smallest values
    difference = (np.max(np_array) - np.min(np_array)) / np.linalg.norm(np_array)
    
    results = [average, std_deviation, difference]

    return results


def embedding_word(word, length = 1):
    array = sa.sentiment(word)
    # 0 word 
    # 1 polarity 
    # 2 subjectivity
    # 3 square sqrt(l-(l/2-p)-(l/2-s))
    # 4 circle 
    square = length - (length/2 - float(array[1]))-(length/2 - float(array[2]))
    if square == 0:
        square = 1
    elif square == -1:
        square = 1j
    elif square < 0:
        square = cmath.sqrt(length - (length/2 - float(array[1]))-(length/2 - float(array[2])))
    else:
        square = math.sqrt(length - (length/2 - float(array[1]))-(length/2 - float(array[2])))
        
    circle = length * (float(array[2])/float(array[1]))
    array[3] = square 
    array[4] = circle
    embed = array    
    return embed

def embedding_word_array(words, length):
    index = 0
    stacked_array = []
    for word in words:
        array = embedding_word(word, len(words))
        array = np.insert(array, 0, index)
        index = index + 1
        stacked_array.append(array)
    stacked_array = np.vstack(stacked_array)
    embed_vector = stacked_array 
    return embed_vector

def embedding_sentence_per_word(sentence):
    if isinstance(sentence, str):
        words = sentence.split()
    words = sentence
    num_words = len(words)
    sentence_text = ""
    polarity = []
    subjectivity = []
    square = []
    circle = [] 
    for word in sentence:
        array = embedding_word(word, num_words)
        polarity.append(array[1])
        subjectivity.append(array[2])
        square.append(array[3])
        circle.append(array[4])
    # circle_stats = calculate_stats(circle)
    # square_stats = calculate_stats(square)
    # polarity_stats = calculate_stats(polarity)
    # subjectivity_stats = calculate_stats(subjectivity)
    # array_2_stack = [polarity_stats, subjectivity_stats, square_stats, circle_stats]
    array_2_stack = [polarity, subjectivity, square, circle]
    stacked_array = np.vstack(array_2_stack)
    return stacked_array

def embedding_sentence(sentences):
    sentences = str(sentences[1:])
    polarity = []
    subjectivity = []
    square = []
    circle = [] 
    length = len(sentences.split())    
    array = embedding_word(sentences, length)
    circle = array[4]
    square = array[3]
    polarity = array[1]
    subjectivity = array[2]
    array_2_stack = [polarity, subjectivity, square, circle]
    stacked_array = np.vstack(array_2_stack)
    
    return stacked_array
    
    
    
    
    
    


