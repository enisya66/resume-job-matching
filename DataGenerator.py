# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:38:47 2019

@author: User
"""

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def cleanup_text(text):
    """
    Preprocess text.
    Args:
        text: raw text
    Returns:
        text: cleaned text in base form and lowercased, without stopwords and punctuation
    """
    # initialize stopwords and stemmer
    stop_words = set(stopwords.words('german'))|set(stopwords.words('english'))
    germanStemmer = SnowballStemmer('german', ignore_stopwords = True)
    # remove punctuation
    text = text.translate(str.maketrans('','',':;,?()[]{}<>"'))
    # stemming
    text = ' '.join(germanStemmer.stem(word) for word in text.split())
    # remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    # lowercase
    return text.lower()

def get_average_text_length(texts): 
    """
    Calculates the average text length in an array/list of texts
    Args:
        texts: 1D array of strings
    Returns:
        the average text length
    """
    return round(sum(len(i) for i in texts) / len(texts), 2)

def plot_text_length(texts):
    lengths = []
    for i in range(len(texts)):
        lengths.append(len(texts[i]))
        
    fig1, ax1 = plt.subplots()
    ax1.set_title('Text Length')
    ax1.boxplot(lengths, vert=False)
    plt.show()
