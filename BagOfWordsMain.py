# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:44:38 2019

@author: User
"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import time

# import functions
from FileReader import generate_data_for_resume_matcher
from DataGenerator import cleanup_text, get_average_text_length
from ModelEvaluation import plot_confusion_matrix, model_classification_report

"""
Bag-of-words method using internal dataset for resume matching
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
x_train, x_test, y_train, y_test = train_test_split(pairs, labels, test_size=TEST_SPLIT, random_state=42)

# start the clock
start = time.time()

# initialize count vectorizer
cvec = CountVectorizer()
similarity = []

# learn the vocabulary dictionary and return term-document matrix
vector = cvec.fit_transform(x_train.ravel()).toarray()

# transform documents to document-term matrix
cv_vector = cvec.transform(x_train[:,0])
job_vector = cvec.transform(x_train[:,1])

print(cv_vector.shape)
print(job_vector.shape)

# calculate cosine similarity
for i in range(x_train.shape[0]):
    similarity.append(cosine_similarity(cv_vector[i,:], job_vector[i,:])[0][0])

similarity = np.array(similarity).reshape(-1,1)
print(similarity.shape)

# visualise data
plt.scatter(similarity, y_train)
plt.show()

# train model
model = RandomForestClassifier(n_estimators=100,max_depth=2).fit(similarity, y_train)

# testing
similarity_test = []
cv_vector_test = cvec.transform(x_test[:,0])
job_vector_test = cvec.transform(x_test[:,1])

# calculate cosine similarity
for i in range(x_test.shape[0]):
    similarity_test.append(cosine_similarity(cv_vector_test[i,:], job_vector_test[i,:])[0][0])

similarity_test = np.array(similarity_test).reshape(-1,1)

# evaluate
y_pred = model.predict(similarity_test)
# print evaluation measures
print(model_classification_report(y_test, y_pred, LABELS))
plot_confusion_matrix(y_test, y_pred, LABELS)
# stop the clock
print('Total time taken: %s seconds' % (time.time() - start))