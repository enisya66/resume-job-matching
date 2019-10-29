# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:12:47 2019

@author: User
"""
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten
from keras.initializers import Constant
from sklearn.pipeline import FeatureUnion
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import time

# import functions
from DataGenerator import cleanup_text, get_average_text_length
from EmbeddingUtils import word_embedding_metadata
from ModelEvaluation import evaluate_continuous_data

# constants
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 300

STS_COLUMNS = ['label','s1','s2']

# read data
sts_train = pd.read_csv('data/sts-train.csv',sep='\t',usecols=[i for i in range(4,7)],names=STS_COLUMNS)
sts_test = pd.read_csv('data/sts-test.csv',sep='\t',usecols=[i for i in range(4,7)],quoting=csv.QUOTE_NONE,names=STS_COLUMNS)

sts_train.head()
sts_test.head()

print('Training size is', sts_train.shape)
print('Test size is', sts_test.shape)

# drop rows with nan values
sts_train = sts_train.dropna(axis=0,how='any')
sts_test = sts_test.dropna(axis=0,how='any')

print('Training size is', sts_train.shape)
print('Test size is', sts_test.shape)

# train model
similarity = []
x_train = sts_train.to_numpy()
y_train = x_train[:,0]

# cleanup text
for x in x_train:
    x[1] = cleanup_text(x[1])
    x[2] = cleanup_text(x[2])

# start the clock
start = time.time()

# generate embedding matrix
tokenizer, embedding_matrix = word_embedding_metadata(x_train.ravel().astype('U'), MAX_NUM_WORDS, EMBEDDING_DIM)

# prepare data for input
s1_sequences = tokenizer.texts_to_sequences(x_train[:,1])
s2_sequences = tokenizer.texts_to_sequences(x_train[:,2])
s1_data = pad_sequences(s1_sequences, maxlen=MAX_SEQUENCE_LENGTH)
s2_data = pad_sequences(s2_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# load pre-trained word embeddings into an Embedding layer (embeddings_initializer)
# note that we set trainable = False so as to keep the embeddings fixed
num_words = len(tokenizer.word_index) + 1
sts_embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# define model
embedding_model = Sequential()
embedding_model.add(sts_embedding_layer)
embedding_model.add(Flatten())
embedding_model.compile('rmsprop', 'mse')

s1_vector = embedding_model.predict(s1_data)
s2_vector = embedding_model.predict(s2_data)

# convert to sparse matrix
s1_vector = sparse.csr_matrix(s1_vector)
s2_vector = sparse.csr_matrix(s2_vector)

# calculate cosine similarity for each pair (not pairwise!)
for i in range(x_train.shape[0]):
    similarity.append(cosine_similarity(s1_vector[i,:], s2_vector[i,:])[0][0])
    
similarity = np.array(similarity).reshape(-1,1)

# visualise data
plt.scatter(similarity, y_train)
plt.show()

# fit model
stsmodel = LinearRegression().fit(similarity, y_train)

# test
similarity_test = []
x_test = sts_test.to_numpy()
y_test = x_test[:,0]

# cleanup text
for x in x_test:
    x[1] = cleanup_text(x[1])
    x[2] = cleanup_text(x[2])
    
# prepare data for input
s1_sequences = tokenizer.texts_to_sequences(x_test[:,1])
s2_sequences = tokenizer.texts_to_sequences(x_test[:,2])
s1_data = pad_sequences(s1_sequences, maxlen=MAX_SEQUENCE_LENGTH)
s2_data = pad_sequences(s2_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# predict
s1_vector_test = embedding_model.predict(s1_data)
s2_vector_test = embedding_model.predict(s2_data)

# convert to sparse matrix
s1_vector_test = sparse.csr_matrix(s1_vector_test)
s2_vector_test = sparse.csr_matrix(s2_vector_test)

# calculate cosine similarity for each pair (not pairwise!)
for i in range(x_test.shape[0]):
    similarity_test.append(cosine_similarity(s1_vector_test[i,:], s2_vector_test[i,:])[0][0])

similarity_test = np.array(similarity_test).reshape(-1,1)

# evaluate
y_pred = stsmodel.predict(similarity_test)

# print evaluation measures
evaluate_continuous_data(y_test, y_pred)
# stop the clock
print('Total time taken: %s seconds' % (time.time() - start))