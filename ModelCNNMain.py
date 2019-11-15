# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:18:26 2019

@author: User
"""

from keras.models import load_model
from keras.layers import Lambda
from keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import keras
import pydot


# import functions
from FileReader import generate_data_for_resume_matcher
from DataGenerator import cleanup_text, get_average_text_length, plot_text_length
from EmbeddingUtils import word_embedding_metadata, create_test_data
from ModelCNN import SiameseBiCNN
from ModelEvaluation import plot_confusion_matrix, model_classification_report, evaluate_continuous_data

# constants
NUM_CLASSES = 5 #len(np.unique(y))
# maximum size of dictionary (no. of unique words)
MAX_NUM_WORDS = 20000
# pre-trained embeddings have a dimension of 300
EMBEDDING_DIM = 300
# texts are padded to this length
# this depends on the average length of a document
MAX_SEQUENCE_LENGTH = 500
VALIDATION_SPLIT = 0.2
RATE_DROP_CNN = 0.2
RATE_DROP_DENSE = 0.25
KERNEL_WIDTH = 3
NUMBER_DENSE_UNITS = 50
ACTIVATION_FUNCTION = 'relu'
LOSS_FUNCTION = 'categorical_crossentropy'
#LOSS_FUNCTION = 'mse'


TEST_SPLIT = 0.2
LABELS = np.array([1,2,3,4,5])

"""
Siamese CNN for resume matching method
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
    
# rescale y value from [5,1] to [0,1]
labels = labels.astype(np.float)
for i in range(len(labels)):
    labels[i] = abs(1-labels[i])/4.0

# print the first 5 pairs/labels
print(pairs[:5,0])
print(pairs[:5,1])
print(labels[:5])

# print the average length of documents
print('Average CV length: ', get_average_text_length(pairs[:,0]))
print('Average job post length: ', get_average_text_length(pairs[:,1]))
plot_text_length(pairs[:,0])
plot_text_length(pairs[:,1])

# print the array shape
print(pairs.shape)
print(labels.shape)

# split data into train and test

tokenizer, embedding_matrix = word_embedding_metadata(pairs, MAX_NUM_WORDS, EMBEDDING_DIM)

# =============================================================================
# cv_sequences = tokenizer.texts_to_sequences(pairs[:,0])
# jobpost_sequences = tokenizer.texts_to_sequences(pairs[:,1])
# cv_data = pad_sequences(cv_sequences, maxlen=MAX_SEQUENCE_LENGTH)
# jobpost_data = pad_sequences(jobpost_sequences, maxlen=MAX_SEQUENCE_LENGTH)
# pairs_sequences = np.concatenate((cv_data, jobpost_data), axis=1)
# =============================================================================

x_train, x_test, y_train, y_test = train_test_split(pairs, labels, stratify=labels, test_size=TEST_SPLIT, random_state=42)

# create model
siamese = SiameseBiCNN(EMBEDDING_DIM , MAX_SEQUENCE_LENGTH, KERNEL_WIDTH , NUMBER_DENSE_UNITS, 
					    RATE_DROP_CNN, RATE_DROP_DENSE, ACTIVATION_FUNCTION, VALIDATION_SPLIT, LOSS_FUNCTION)

# normalize labels to be used with to_categorical
encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)

y_train = to_categorical(y_train)

# create_train_data not compatible with train_test_split
model = siamese.train_model(x_train, y_train, tokenizer, embedding_matrix, model_save_directory='./models/cnn/')

# testing
#best_model_path = './models/lstm/checkpoints/1572426219/lstm_50_50_0.20_0.25.h5'
#model = load_model(best_model_path, custom_objects={'Lambda': Lambda}, compile=False)

# visualise model
keras.utils.vis_utils.pydot = pydot
plot_model(model, show_shapes=True)

# cleanup text
for x in x_test:
    x[0] = cleanup_text(x[0])
    x[1] = cleanup_text(x[1])

test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,x_test, MAX_SEQUENCE_LENGTH)
y_pred_arr = model.predict([test_data_x1, test_data_x2], verbose=1)
#y_pred = y_pred[:,0]

# get the class with max value
y_pred = y_pred_arr.argmax(1)
# normalize labels to be used with to_categorical
y_test = encoder.transform(y_test)

# print evaluation measures
print(model_classification_report(y_test, y_pred, LABELS))
plot_confusion_matrix(y_test, y_pred, LABELS)
#evaluate_continuous_data(y_test, y_pred)