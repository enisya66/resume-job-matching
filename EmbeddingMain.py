# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:07:37 2019

@author: User
"""
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np
import time

# import functions
from FileReader import generate_data_for_resume_matcher
from DataGenerator import cleanup_text, get_average_text_length
from EmbeddingUtils import word_embedding_metadata
from ModelEvaluation import plot_confusion_matrix, model_classification_report

# constants
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 300

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

# start the clock
start = time.time()

# generate embedding matrix
tokenizer, embedding_matrix = word_embedding_metadata(pairs, MAX_NUM_WORDS, EMBEDDING_DIM)

# prepare input data
cv_sequences = tokenizer.texts_to_sequences(pairs[:,0])
jobpost_sequences = tokenizer.texts_to_sequences(pairs[:,1])
cv_data = pad_sequences(cv_sequences, maxlen=MAX_SEQUENCE_LENGTH)
jobpost_data = pad_sequences(jobpost_sequences, maxlen=MAX_SEQUENCE_LENGTH)
pairs_sequences = np.concatenate((cv_data, jobpost_data), axis=1)

print(cv_data.shape)
print(jobpost_data.shape)
print(pairs_sequences.shape)

x_train, x_test, y_train, y_test = train_test_split(pairs_sequences, labels, test_size=TEST_SPLIT, random_state=42)

print(x_train.shape)

# load pre-trained word embeddings into an Embedding layer (embeddings_initializer)
# note that we set trainable = False so as to keep the embeddings fixed
num_words = len(tokenizer.word_index) + 1
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# print embedding shape
print('Embedding shape', embedding_layer.input_dim)
print(embedding_layer.output_dim)

# define model
embedding_model = Sequential()
embedding_model.add(embedding_layer)
embedding_model.add(Flatten())
# 1st param optimizer: rmsprop, adadelta, adam, adamax
# 2nd param loss: mse, categorical_crossentropy
# in this case as embedding layer is 'frozen' optimizer and loss are not necessary?
embedding_model.compile('rmsprop', 'mse')

cv_embeddings = embedding_model.predict(x_train[:,0:MAX_SEQUENCE_LENGTH])
jobpost_embeddings = embedding_model.predict(x_train[:,MAX_SEQUENCE_LENGTH:MAX_SEQUENCE_LENGTH*2])
print('embedding cv shape', cv_embeddings.shape)
print('embedding jobpost shape', jobpost_embeddings.shape)

# convert to sparse matrix
cv_embeddings = sparse.csr_matrix(cv_embeddings)
jobpost_embeddings = sparse.csr_matrix(jobpost_embeddings)

# from here copied from BoW method

# calculate cosine similarity
similarity = []
for i in range(x_train.shape[0]):
    similarity.append(cosine_similarity(cv_embeddings[i,:], jobpost_embeddings[i,:])[0][0])

similarity = np.array(similarity).reshape(-1,1)
print(similarity.shape)

# visualise data
plt.scatter(similarity, y_train)
plt.show()

# train model
# did not work with Naive Bayes as cosine similarity has negative values
model = RandomForestClassifier(n_estimators=100,max_depth=2).fit(similarity, y_train)

#test
similarity_test = []
cv_embeddings_test = embedding_model.predict(x_test[:,0:MAX_SEQUENCE_LENGTH])
jobpost_embeddings_test = embedding_model.predict(x_test[:,MAX_SEQUENCE_LENGTH:MAX_SEQUENCE_LENGTH*2])

# convert to sparse matrix
cv_embeddings_test = sparse.csr_matrix(cv_embeddings_test)
jobpost_embeddings_test = sparse.csr_matrix(jobpost_embeddings_test)

# calculate cosine similarity
for i in range(x_test.shape[0]):
    similarity_test.append(cosine_similarity(cv_embeddings_test[i,:], jobpost_embeddings_test[i,:])[0][0])

similarity_test = np.array(similarity_test).reshape(-1,1)

# evaluate
y_pred = model.predict(similarity_test)

# print evaluation measures
print(model_classification_report(y_test, y_pred, LABELS))
plot_confusion_matrix(y_test, y_pred, LABELS)

# stop the clock
print('Total time taken: %s seconds' % (time.time() - start))