# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:12:47 2019

@author: User
"""
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize
from keras.layers import Embedding, Flatten, Input, Lambda, Dense
from keras.initializers import Constant
from keras import backend as K
import fasttext
from sklearn.pipeline import FeatureUnion
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from scipy import sparse
from scipy.spatial.distance import cosine
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
MAX_SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 300

STS_COLUMNS = ['label','s1','s2']

def cosine_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    print('shape1',shape1)
    print('shape2',shape2)
    return (shape1[0], 1)

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
    x[1] = cleanup_text(x[1], False)
    x[2] = cleanup_text(x[2], False)

# start the clock
start = time.time()

# generate embedding matrix
#tokenizer, embedding_matrix = word_embedding_metadata(x_train.ravel().astype('U'), MAX_NUM_WORDS, EMBEDDING_DIM)

# prepare data for input


s1_sequences = [word_tokenize(i) for i in x_train[:,1]]
s2_sequences = [word_tokenize(i) for i in x_train[:,2]]
#s1_data = pad_sequences(s1_sequences, maxlen=MAX_SEQUENCE_LENGTH)
#s2_data = pad_sequences(s2_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# load pre-trained word embeddings into an Embedding layer (embeddings_initializer)
# note that we set trainable = False so as to keep the embeddings fixed
#num_words = len(tokenizer.word_index) + 1
#sts_embedding_layer = Embedding(num_words,
#                            EMBEDDING_DIM,
#                            embeddings_initializer=Constant(embedding_matrix),
#                            input_length=MAX_SEQUENCE_LENGTH,
#                            trainable=False)

# define model

model = fasttext.load_model('cc.en.300.bin')

for i in range(len(s1_sequences)):
    s1_vector = np.mean([model[word] for word in s1_sequences[i]], axis=0)
    s2_vector = np.mean([model[word] for word in s2_sequences[i]], axis=0)
    similarity.append(cosine(s1_vector, s2_vector))


#s1_vector = embedding_model.predict(s1_data)
#s2_vector = embedding_model.predict(s2_data)

# convert to sparse matrix
#s1_vector = sparse.csr_matrix(s1_vector)
#s2_vector = sparse.csr_matrix(s2_vector)

# calculate cosine similarity for each pair (not pairwise!)
#for i in range(x_train.shape[0]):
#    similarity.append(cosine_similarity(s1_vector[i,:,:], s2_vector[i,:,:])[0][0])
    
similarity = np.array(similarity).reshape(-1,1)
# fill in nan values due to empty sentences with 0
similarity = np.nan_to_num(similarity)
y_train = y_train.astype('int')
# visualise data
plt.scatter(similarity, y_train)
plt.show()

# fit model
stsmodel = LinearRegression().fit(similarity, y_train)

del s1_vector
del s2_vector
# test
similarity_test = []
x_test = sts_test.to_numpy()
y_test = x_test[:,0]

# cleanup text
for x in x_test:
    x[1] = cleanup_text(x[1])
    x[2] = cleanup_text(x[2])
    
# prepare data for input
s1_sequences = [word_tokenize(i) for i in x_test[:,1]]
s2_sequences = [word_tokenize(i) for i in x_test[:,2]]
#s1_data = pad_sequences(s1_sequences, maxlen=MAX_SEQUENCE_LENGTH)
#s2_data = pad_sequences(s2_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# predict
#s1_vector_test = [model[x] for word in s1_sequences for x in word]
#s2_vector_test = [model[x] for word in s2_sequences for x in word]



# convert to sparse matrix
#s1_vector_test = sparse.csr_matrix(s1_vector_test)
#s2_vector_test = sparse.csr_matrix(s2_vector_test)

for i in range(len(s1_sequences)):
    s1_vector = np.mean([model[word] for word in s1_sequences[i]], axis=0)
    s2_vector = np.mean([model[word] for word in s2_sequences[i]], axis=0)
    similarity_test.append(cosine(s1_vector, s2_vector))

# calculate cosine similarity for each pair (not pairwise!)
#for i in range(x_test.shape[0]):
#    similarity_test.append(cosine_similarity(s1_vector_test[i,:], s2_vector_test[i,:])[0][0])

similarity_test = np.array(similarity_test).reshape(-1,1)
similarity_test = np.nan_to_num(similarity_test)
y_test = y_test.astype('int')


# evaluate
y_pred = stsmodel.predict(similarity_test)


# print evaluation measures
evaluate_continuous_data(y_test, y_pred)
# stop the clock
print('Total time taken: %s seconds' % (time.time() - start))