# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:15:14 2019

@author: User
"""
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from fasttext import load_model
from tqdm import tqdm
import random
import numpy as np
import os
import gc # garbage collector

def load_pretrained_embeddings():
    """
    Loads pre-trained word vectors
    Returns:
        word vectors
    """
    embeddings_index = {}
# =============================================================================
#     model_path = datapath('cc.en.300.bin')
#     embeddings_index = load_facebook_vectors(model_path)
# =============================================================================
    with open(os.path.join('', 'crawl-300d-2M.vec'), encoding='utf8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    
    print('Embedding data loaded')
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def create_embedding_matrix(tokenizer, max_num_words, embedding_dim):
    """
    Create embedding matrix containing word indexes and respective vectors from word vectors
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object containing word indexes
        word_vectors (dict): dict containing word and their respective vectors
        max_num_words (int): maximum number of words
        embedding_dim (int): dimension of word vector
    Returns:
        embedding matrix
    """
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    num_words = min(max_num_words, len(word_index) + 1)
    print('Number of words', num_words)
    
    embedding_matrix = np.zeros((num_words, embedding_dim))
    
    words_not_found = []
    
    model = load_model('cc.en.300.bin')
    
    # technically should also produce word vector for out of vocab words.
    with tqdm(total=num_words, desc='Embeddings', unit=' words') as pbar:
        for word, i in word_index.items():
            if i >= max_num_words:
                continue
            embedding_vector = model.get_word_vector(word).astype('float32')
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
                pbar.update()
            else:
                words_not_found.append(word)
                pbar.update()
            
    
    print('total words not found: ', len(words_not_found))
     
    return embedding_matrix
    
def word_embedding_metadata(documents, max_num_words, embedding_dim):
    """
    Args:
        documents: (n-dimensional) array containing data
        max_num_words (int): maximum number of words
        embedding_dim: embedding dimension
    Returns:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        embedding_matrix (dict): dict with word_index and vector mapping
    """
    tokenizer = Tokenizer(num_words=max_num_words)
    
    # multi dimensional data
    if(documents.ndim > 1):
        tokenizer.fit_on_texts(documents.ravel())
    else:
        tokenizer.fit_on_texts(documents)
    
    #word_vector = load_pretrained_embeddings()
    embedding_matrix = create_embedding_matrix(tokenizer, max_num_words, embedding_dim)
    
    # clear cache
    #del word_vector
    #gc.collect()
    return tokenizer, embedding_matrix
    
def create_train_dev_set(tokenizer, sentences_pair, is_similar, max_sequence_length, validation_split_ratio):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        sentences_pair (list): list of tuple of sentences pairs (Training)
        is_similar (list): list containing labels if respective sentences in sentence1 and sentence2
                           are same or not (1 if same else 0)
        max_sequence_length (int): max sequence length of sentences to apply padding
        validation_split_ratio (float): contain ratio to split training data into validation data
    Returns:
        train_data_1 (list): list of input features for training set from sentences1
        train_data_2 (list): list of input features for training set from sentences2
        labels_train (np.array): array containing similarity score for training data
        leaks_train(np.array): array of training leaks features
        val_data_1 (list): list of input features for validation set from sentences1
        val_data_2 (list): list of input features for validation set from sentences1
        labels_val (np.array): array containing similarity score for validation data
        leaks_val (np.array): array of validation leaks features
    """
    sentences1 = [x[0] for x in sentences_pair]
    sentences2 = [x[1] for x in sentences_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(train_sequences_1, train_sequences_2)]

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)
    train_labels = np.array(is_similar)
    leaks = np.array(leaks)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1
    del train_padded_data_2
    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, leaks_train, val_data_1, val_data_2, labels_val, leaks_val

def build_dataset(tokenizer, sentences_pair, is_similar, max_sequence_length, validation_split_ratio):
        
    sentences1 = [x[0] for x in sentences_pair]
    sentences2 = [x[1] for x in sentences_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)
    train_labels = np.array(is_similar)
    
    # randomly select from train for dev
    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

def create_test_data(tokenizer, test_sentences_pair, max_sequence_length):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        test_sentences_pair (list): list of tuple of sentences pairs
        max_sequence_length (int): max sequence length of sentences to apply padding
    Returns:
        test_data_1 (list): list of input features for training set from sentences1
        test_data_2 (list): list of input features for training set from sentences2
    """
    test_sentences1 = [x[0] for x in test_sentences_pair]
    test_sentences2 = [x[1] for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                  for x1, x2 in zip(test_sequences_1, test_sequences_2)]

    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)

    return test_data_1, test_data_2, leaks_test