# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:21:26 2019

@author: User
"""

from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors
from gensim.scripts.glove2word2vec import glove2word2vec

from gensim.models import FastText

"""
Test program to include prediction of out-of-vocabulary words.
"""

vector_path = datapath(r'C:\Users\User\Documents\Python\Bachelor Thesis\glove.42B.300d.w2vformat.txt')
#fasttext_vector = load_facebook_vectors(vector_path)
#
## test
#
#test1 = fasttext_vector.most_similar(positive=['Ansible'])
#print(test1)
