# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:50:39 2019

@author: User
"""

"""
An interface which gives a list of predictions given a list of text pairs.
"""

from keras.models import load_model
import glob
import pickle
import os
import csv

from FileReader import read_file
from EmbeddingUtils import create_test_data

MAX_SEQUENCE_LENGTH = 12

# load latest model
list_of_files = glob.glob('/models/*.h5') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print('Model ', latest_file)

model = load_model(latest_file)

# load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# get predictions as .csv file
x_test = []

with open('predict.csv', 'rt') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        next(csvReader)
        for row in csvReader:
            cv_text = read_file('test_cv/', row[0])
            jobpost_text = read_file('test_jobpost/', row[1])
           
            x_test += [[cv_text, jobpost_text]]

# preprocess test data to a padded sequence of integers
test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,x_test,MAX_SEQUENCE_LENGTH)

# predict
y_pred_arr = model.predict([test_data_x1, test_data_x2], verbose=1)

# evaluate
# get the class with max value
y_pred = y_pred_arr.argmax(1)

# TODO return y_pred array in a REST API