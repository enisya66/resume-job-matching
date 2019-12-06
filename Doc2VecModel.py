# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:12:48 2019

@author: User
"""

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np


from FileReader import generate_data_for_resume_matcher
from DataGenerator import cleanup_text, get_average_text_length, plot_text_length
from ModelEvaluation import plot_confusion_matrix, model_classification_report


TEST_SPLIT = 0.2
LABELS = np.array([1,2,3,4,5])

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
plot_text_length(pairs[:,0])
plot_text_length(pairs[:,1])

# print the array shape
print(pairs.shape)
print(labels.shape)

x_train, x_test, y_train, y_test = train_test_split(pairs, labels, test_size=TEST_SPLIT, random_state=42)

questions1 = x_train[:,0]
questions2 = x_train[:,1]


questions1_split = []
for question in questions1:
    questions1_split.append(question.split())
    
questions2_split = []
for question in questions2:
    questions2_split.append(question.split())

# Contains the processed questions for Doc2Vec
questions_labeled = []

for i in range(len(questions1)):
    # Question strings need to be separated into words
    # Each question needs a unique label
    questions_labeled.append(questions1[i].split())
    questions_labeled.append(questions2[i].split())

questions_labeled = iter(questions_labeled)
model = Doc2Vec(min_count=1, sample=1e-4)
sentences = [TaggedDocument(sentence, 'tag') for sentence in questions_labeled]
model.build_vocab(questions_labeled)

for epoch in range(5):
    model.train(questions_labeled)

doc2vec_scores = []
for i in range(len(questions1_split)):
    # n_similarity computes the cosine similarity in Doc2Vec
    score = model.n_similarity(questions1_split[i],questions2_split[i])
    doc2vec_scores.append(score)

# train
classifier = LogisticRegression().fit(doc2vec_scores, y_train)

# test
# evaluate

questions1 = x_test[:,0]
questions2 = x_test[:,1]


questions1_split = []
for question in questions1:
    questions1_split.append(question.split())
    
questions2_split = []
for question in questions2:
    questions2_split.append(question.split())
    
similarity_test = []
for i in range(len(questions1_split)):
    # n_similarity computes the cosine similarity in Doc2Vec
    score = model.n_similarity(questions1_split[i],questions2_split[i])
    similarity_test.append(score)

y_pred = classifier.predict(similarity_test)

# print evaluation measures
print(model_classification_report(y_test, y_pred, LABELS))
plot_confusion_matrix(y_test, y_pred, LABELS)