# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:24:53 2019

@author: User
"""

from keras.models import load_model
from keras.utils import plot_model
import pandas as pd
import csv

# import functions
from FileReader import generate_data_for_sts
from DataGenerator import cleanup_text
from EmbeddingUtils import word_embedding_metadata, create_test_data
from ModelCNN import SiameseCNN
from ModelCNNMultiFilter import SiameseMultiCNN
from ModelEvaluation import evaluate_continuous_data

# constants

# maximum size of dictionary (no. of unique words)
MAX_NUM_WORDS = 20000
# pre-trained embeddings have a dimension of 300
EMBEDDING_DIM = 300
# texts are padded to this length
# this depends on the average length of a document/text
MAX_SEQUENCE_LENGTH = 12
VALIDATION_SPLIT = 0.2
RATE_DROP_CNN = 0.2
RATE_DROP_DENSE = 0.5
KERNEL_WIDTH = 3
NUMBER_DENSE_UNITS = 50
ACTIVATION_FUNCTION = 'relu'
LOSS_FUNCTION = 'mse'

STS_COLUMNS = ['label','s1','s2']

# =============================================================================
# sts_train = pd.read_csv('data/sts-train.csv',sep='\t',usecols=[i for i in range(4,7)],names=STS_COLUMNS)
# sts_test = pd.read_csv('data/sts-test.csv',sep='\t',usecols=[i for i in range(4,7)],quoting=csv.QUOTE_NONE,names=STS_COLUMNS)
# =============================================================================
sts_other = pd.read_csv('data/sts-other.csv',sep='\t',usecols=[i for i in range(3,6)],names=STS_COLUMNS)
sts_mt = pd.read_csv('data/sts-mt.csv',sep='\t',usecols=[i for i in range(3,6)],names=STS_COLUMNS)
sts_dev = pd.read_csv('data/sts-dev.csv',sep='\t',usecols=[i for i in range(4,7)],names=STS_COLUMNS)

sts_train, sts_test = generate_data_for_sts()

# trial data augmentation
sts_train = sts_train.append(sts_other, ignore_index=True)
#sts_train = sts_train.append(sts_dev, ignore_index=True)
sts_test = sts_test.append(sts_mt, ignore_index=True)

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
#for x in x_train:
#    x[0] = cleanup_text(x[0])
#    x[1] = cleanup_text(x[1])
# 

# rescale y value from [0,5] to [0,1]
for i in range(len(y_train)):
    y_train[i] = y_train[i]/5


# # generate embedding matrix
tokenizer, embedding_matrix = word_embedding_metadata(x_train.ravel().astype('U'), MAX_NUM_WORDS, EMBEDDING_DIM)
# 
# create model
siamese = SiameseMultiCNN(EMBEDDING_DIM , MAX_SEQUENCE_LENGTH, KERNEL_WIDTH , NUMBER_DENSE_UNITS, 
					    RATE_DROP_CNN, RATE_DROP_DENSE, ACTIVATION_FUNCTION, VALIDATION_SPLIT, LOSS_FUNCTION)

 
model = siamese.train_model(x_train, y_train, tokenizer, embedding_matrix, model_save_directory='./models/')
 
# =============================================================================

# testing
#best_model_path='./models/checkpoints/1571869395/lstm_50_50_0.17_0.25.h5'
#model = load_model(best_model_path)

# visualise model
#plot_model(model)

# evaluate
x_test = sts_test.to_numpy()
y_test = x_test[:,0]
x_test = x_test[:,[1,2]]

# cleanup text
#for x in x_test:
#    x[0] = cleanup_text(x[0])
#    x[1] = cleanup_text(x[1])
    
    
# rescale y value from [0,5] to [0,1]
for i in range(len(y_test)):
    y_test[i] = y_test[i]/5 

test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,x_test, MAX_SEQUENCE_LENGTH)
y_pred = model.predict([test_data_x1, test_data_x2], verbose=1)
y_pred = y_pred[:,0]

# print evaluation measures
evaluate_continuous_data(y_test, y_pred)