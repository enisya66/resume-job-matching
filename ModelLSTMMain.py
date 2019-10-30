# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:59:24 2019

@author: User
"""
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import keras
import pydot
import time
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# import functions
from FileReader import generate_data_for_resume_matcher
from DataGenerator import cleanup_text, get_average_text_length
from EmbeddingUtils import word_embedding_metadata, create_test_data
from ModelLSTM import SiameseBiLSTM
from ModelEvaluation import plot_confusion_matrix, model_classification_report, save_predictions

# constants
NUM_CLASSES = 5 #len(np.unique(y))
MAX_NUM_WORDS = 20000
# TODO embedding dimension depends on word vector?
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 500
VALIDATION_SPLIT = 0.2
RATE_DROP_LSTM = 0.2
RATE_DROP_DENSE = 0.25
NUMBER_LSTM = 50
NUMBER_DENSE_UNITS = 50
ACTIVATION_FUNCTION = 'relu'
LOSS_FUNCTION = 'categorical_crossentropy'

"""
Siamese LSTM for resume matching method
"""

# load data for resume matcher
pairs, labels = generate_data_for_resume_matcher('data.csv')

# visualise data
# print the first 5 pairs/labels
print(pairs[:5])
print(labels[:5])

# print the average length of documents
print('Average CV length: ', get_average_text_length(pairs[:,0]))
print('Average job post length: ', get_average_text_length(pairs[:,1]))

# cleanup text
for p in pairs:
    p[0] = cleanup_text(p[0])
    p[1] = cleanup_text(p[1])
    
# print the first 5 pairs/labels
print(pairs[:5])
print(labels[:5])

# print the average length of documents
print('Average CV length: ', get_average_text_length(pairs[:,0]))
print('Average job post length: ', get_average_text_length(pairs[:,1]))

# print the array shape
print(pairs.shape)
print(labels.shape)

# split data into train and test
TEST_SPLIT = 0.25
LABELS = np.array([1,2,3,4,5])

tokenizer, embedding_matrix = word_embedding_metadata(pairs, MAX_NUM_WORDS, EMBEDDING_DIM)

# =============================================================================
# cv_sequences = tokenizer.texts_to_sequences(pairs[:,0])
# jobpost_sequences = tokenizer.texts_to_sequences(pairs[:,1])
# cv_data = pad_sequences(cv_sequences, maxlen=MAX_SEQUENCE_LENGTH)
# jobpost_data = pad_sequences(jobpost_sequences, maxlen=MAX_SEQUENCE_LENGTH)
# pairs_sequences = np.concatenate((cv_data, jobpost_data), axis=1)
# =============================================================================

x_train, x_test, y_train, y_test = train_test_split(pairs, labels, test_size=TEST_SPLIT, random_state=42)

# create model
siamese = SiameseBiLSTM(EMBEDDING_DIM , MAX_SEQUENCE_LENGTH, NUMBER_LSTM , NUMBER_DENSE_UNITS, 
					    RATE_DROP_LSTM, RATE_DROP_DENSE, ACTIVATION_FUNCTION, VALIDATION_SPLIT, LOSS_FUNCTION)

# normalize labels to be used with to_categorical
encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)

y_train = to_categorical(y_train)

# create_train_data not compatible with train_test_split
best_model_path = siamese.train_model(x_train, y_train, tokenizer, embedding_matrix, model_save_directory='./models/lstm/')

# testing
#best_model_path = './models/lstm/checkpoints/1572426219/lstm_50_50_0.20_0.25.h5'
model = load_model(best_model_path)

# visualise model
keras.utils.vis_utils.pydot = pydot
plot_model(model)

# cleanup text
for x in x_test:
    x[0] = cleanup_text(x[0])
    x[1] = cleanup_text(x[1])

test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,x_test, MAX_SEQUENCE_LENGTH)
y_pred = model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1)

# get the class with max value
y_pred = y_pred.argmax(1)
# normalize labels to be used with to_categorical
y_test = encoder.transform(y_test)

# print evaluation measures
print(model_classification_report(y_test, y_pred, LABELS))
plot_confusion_matrix(y_test, y_pred, LABELS)
