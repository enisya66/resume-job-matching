# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:58:32 2019

@author: User
"""

"""
Bag-of-words method using STS dataset for semantic text similarity task
"""
# note that unlike the previous dataset, the label for this set is continuous in the range [0-5]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import time

# import functions
from DataGenerator import cleanup_text, get_average_text_length
from ModelEvaluation import evaluate_continuous_data

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

# learn vocabulary
cvec1 = CountVectorizer()
vector = cvec1.fit_transform((x_train[:,[1,2]].ravel()).astype('U'))
s1_vector = cvec1.transform(x_train[:,1])
s2_vector = cvec1.transform(x_train[:,2])

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

s1_vector_test = cvec1.transform(x_test[:,1])
s2_vector_test = cvec1.transform(x_test[:,2])

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