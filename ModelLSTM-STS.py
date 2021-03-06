# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:30:38 2019

@author: User
"""

from keras.models import load_model
from keras.utils import plot_model
import pandas as pd
import numpy as np
import csv
from statsmodels.nonparametric.kernel_regression import KernelReg

# import functions
from FileReader import generate_data_for_sts
from DataGenerator import cleanup_text
from EmbeddingUtils import word_embedding_metadata, create_test_data
#from ModelLSTM import SiameseBiLSTM
from ModelMaLSTM import SiameseMaLSTM
from ModelEvaluation import evaluate_continuous_data

# constants
MAX_NUM_WORDS = 20000
# TODO embedding dimension depends on word vector?
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 12
VALIDATION_SPLIT = 0.2
RATE_DROP_LSTM = 0.2
RATE_DROP_DENSE = 0.25
NUMBER_LSTM = 50
NUMBER_DENSE_UNITS = 64
LEARNING_RATE = 0.001
ACTIVATION_FUNCTION = 'relu'
LOSS_FUNCTION = 'mse'

sts_train, sts_test = generate_data_for_sts()

x_train = sts_train.to_numpy()
y_train = x_train[:,0]
x_train = x_train[:,[1,2]]
# 
# print(x_train)
# # cleanup text
#for x in x_train:
#    x[0] = cleanup_text(x[0], False)
#    x[1] = cleanup_text(x[1], False)

# rescale y value from [0,5] to [0,1]
for i in range(len(y_train)):
    y_train[i] = y_train[i]/5
# 
# # generate embedding matrix
tokenizer, embedding_matrix = word_embedding_metadata(x_train.ravel().astype('U'), MAX_NUM_WORDS, EMBEDDING_DIM)
# 
# create model
siamese = SiameseMaLSTM(EMBEDDING_DIM , MAX_SEQUENCE_LENGTH, NUMBER_LSTM , NUMBER_DENSE_UNITS, 
 					    RATE_DROP_LSTM, RATE_DROP_DENSE, LEARNING_RATE,
                        ACTIVATION_FUNCTION, VALIDATION_SPLIT, LOSS_FUNCTION)
 
 
model = siamese.train_model(x_train, y_train, tokenizer, embedding_matrix, model_save_directory='./models/')
 
# =============================================================================

# testing
#best_model_path='./models/checkpoints/1571869395/lstm_50_50_0.17_0.25.h5'
#model = load_model(best_model_path)

# visualise model
plot_model(model)

# evaluate
x_test = sts_test.to_numpy()
y_test = x_test[:,0]
x_test = x_test[:,[1,2]]

# cleanup text
#for x in x_test:
#    x[0] = cleanup_text(x[0], False)
#    x[1] = cleanup_text(x[1], False)

# rescale y value from [0,5] to [0,1]
for i in range(len(y_test)):
    y_test[i] = y_test[i]/5
    

test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,x_test, MAX_SEQUENCE_LENGTH)
#y_pred = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
y_pred = model.predict([test_data_x1, test_data_x2], verbose=1)
y_pred = y_pred[:,0]

# test nonparametric regression
# =============================================================================
# y = np.append(y_train, y_test, axis=1)
# X = np.append(x_train, x_test, axis=1)
# npr = KernelReg(endog=y, exog=X, var_type='c')
# mean, _ = npr.fit()
# print(mean)
# =============================================================================


# convert back y value from [0,1] to [0,5]
for i in range(len(y_test)):
    y_test[i] = y_test[i]*5
    y_pred[i] = y_pred[i]*5

# print evaluation measures
evaluate_continuous_data(y_test, y_pred)
