# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:30:38 2019

@author: User
"""

from keras.models import load_model
from keras.utils import plot_model
import pandas as pd
import csv
import time

# import functions
from DataGenerator import cleanup_text, get_average_text_length
from EmbeddingUtils import word_embedding_metadata, create_test_data
from ModelLSTM import SiameseBiLSTM
from ModelEvaluation import evaluate_continuous_data

# constants
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
LOSS_FUNCTION = 'mse'

STS_COLUMNS = ['label','s1','s2']

# =============================================================================
# # read data
sts_train = pd.read_csv('data/sts-train.csv',sep='\t',usecols=[i for i in range(4,7)],names=STS_COLUMNS)
sts_test = pd.read_csv('data/sts-test.csv',sep='\t',usecols=[i for i in range(4,7)],quoting=csv.QUOTE_NONE,names=STS_COLUMNS)
# 
sts_train.head()
sts_test.head()
# 
# print('Training size is', sts_train.shape)
# print('Test size is', sts_test.shape)
# 
# # drop rows with nan values
sts_train = sts_train.dropna(axis=0,how='any')
sts_test = sts_test.dropna(axis=0,how='any')
# 
# print('Training size is', sts_train.shape)
# print('Test size is', sts_test.shape)
# 
# # train model
x_train = sts_train.to_numpy()
y_train = x_train[:,0]
x_train = x_train[:,[1,2]]
# 
# print(x_train)
# # cleanup text
for x in x_train:
    x[0] = cleanup_text(x[0])
    x[1] = cleanup_text(x[1])
# 
# 
# # generate embedding matrix
tokenizer, embedding_matrix = word_embedding_metadata(x_train.ravel().astype('U'), MAX_NUM_WORDS, EMBEDDING_DIM)
# 
# # create model
# siamese = SiameseBiLSTM(EMBEDDING_DIM , MAX_SEQUENCE_LENGTH, NUMBER_LSTM , NUMBER_DENSE_UNITS, 
# 					    RATE_DROP_LSTM, RATE_DROP_DENSE, ACTIVATION_FUNCTION, VALIDATION_SPLIT, LOSS_FUNCTION)
# 
# # start the clock
# start = time.time()
# 
# best_model_path = siamese.train_model(x_train, y_train, tokenizer, embedding_matrix, model_save_directory='./models/')
# 
# # stop the clock
# print('Total time taken: %s seconds' % (time.time() - start))
# =============================================================================

# testing
best_model_path='./models/checkpoints/1571869395/lstm_50_50_0.17_0.25.h5'
model = load_model(best_model_path)

# visualise model
plot_model(model)

# evaluate
x_test = sts_test.to_numpy()
y_test = x_test[:,0]
x_test = x_test[:,[1,2]]

# cleanup text
for x in x_test:
    x[0] = cleanup_text(x[0])
    x[1] = cleanup_text(x[1])

test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,x_test, MAX_SEQUENCE_LENGTH)
y_pred = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())

# print evaluation measures
evaluate_continuous_data(y_test, y_pred)
