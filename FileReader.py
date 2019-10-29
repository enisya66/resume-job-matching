# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:05:38 2019

@author: User
"""

# prerequisites
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download()

import numpy as np
import os
import csv

def read_file(folder, file):
    """
    Helper method that reads a .txt file and returns the text body
    Args:
        folder: path to directory
        file: filename
    Returns:
        data: content of the file as string
    """
    
    data = ''
    if os.path.isfile(os.path.join(folder, file + '.txt')):
        filename = os.path.join(folder, file)
        with open(filename + '.txt', 'r', encoding='utf8') as file:
            data = file.read().replace('\n', ' ')
    return data
    
# TODO something on imbalanced classes
# TODO should this be stored somewhere?
def generate_data_for_resume_matcher(filename):
    """
    Create array with .csv file for mapping as input
    Args:
        filename: name of the .csv file
    Returns:
        pairs: sentence pairs [s1,s2] as numpy array
        labels: integer numbers as numpy array
    """
    pairs = []
    labels = []
    
    with open(filename, 'rt') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        next(csvReader)
        for row in csvReader:
            cv_text = read_file('cv/', row[0])
            jobpost_text = read_file('jobpost/', row[1])
            
            pairs += [[cv_text, jobpost_text]]
            labels.append(int(row[2]))
    
    return np.array(pairs), np.transpose(labels)
