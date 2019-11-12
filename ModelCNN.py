# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:55:35 2019

@author: User
"""

# TODO after ModelLSTM runs, adjust LSTM layers to CNN
# keras imports
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Lambda, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model, Sequential
from keras import backend as K
from sklearn.utils.class_weight import compute_class_weight

# std imports
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# TODO create train dev set w/o leaks
from EmbeddingUtils import create_train_dev_set


class SiameseBiCNN:
    def __init__(self, embedding_dim, max_sequence_length, kernel_width, number_dense, rate_drop_cnn, 
                 rate_drop_dense, hidden_activation, validation_split_ratio, loss_function):
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.kernel_width = kernel_width
        self.rate_drop_cnn = rate_drop_cnn
        self.number_dense_units = number_dense
        self.activation_function = hidden_activation
        self.rate_drop_dense = rate_drop_dense
        self.validation_split_ratio = validation_split_ratio
        self.loss_function = loss_function

    
    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))
    
    def manhattan_distance(self, vects):
        x, y = vects
        return K.sum(K.abs(x - y), axis=1, keepdims=True)
    
    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        print('shape1',shape1)
        print('shape2',shape2)
        return (shape1[0], 1)
    
    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    def init_weights(self, shape, name=None):
        values = np.random.normal(loc=0,scale=1e-2,size=shape)
        return K.variable(values, name=name)

    def train_model(self, sentences_pair, categories, tokenizer, embedding_matrix, model_save_directory='./models/'):
        """
        Train Siamese network to find similarity between sentences in `sentences_pair`
            Steps Involved:
                1. Pass each pair from sentences_pairs to CNN.
                2. Merge the vectors from CNN and pass to dense layer.
                3. Pass the dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            sentences_pair (list): list of tuple of sentence pairs
            categories (list): target values (1-5)
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix
            model_save_directory (str): working directory for where to save models
        Returns:
            return (best_model_path):  path of best model
        """
        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, sentences_pair,
                                                                               categories, self.max_sequence_length,
                                                                               self.validation_split_ratio)

        if train_data_x1 is None:
            print("++++ !! Failure: Unable to train model ++++")
            return None

        #class_weight = compute_class_weight()
        
        nb_words = len(tokenizer.word_index) + 1

        # Creating word embedding layer
        embedding_layer = Embedding(nb_words, self.embedding_dim, weights=[embedding_matrix],
                                    input_length=self.max_sequence_length, trainable=False)


        
        # CNN base network
        # TODO use bias?
        # TODO multi filters + concat
        cnn_layer = Sequential([Conv1D(64, self.kernel_width, activation=self.activation_function),
                                MaxPooling1D(2),
                                Dropout(self.rate_drop_cnn),
                                Conv1D(128, self.kernel_width, activation=self.activation_function),
                                MaxPooling1D(2),
                                Dropout(self.rate_drop_cnn),
                                Conv1D(256, self.kernel_width, activation=self.activation_function),
                                GlobalMaxPooling1D(),
                                Dropout(self.rate_drop_dense)
                                #Dense(2048, activation=self.activation_function, kernel_regularizer=l2(1e-3))
                                ])
    
# =============================================================================
#         cnn_layer = Sequential([Conv1D(64, self.kernel_width, activation=self.activation_function),
#                                     MaxPooling1D(2),
#                                     Dropout(self.rate_drop_cnn),
#                                     Conv1D(128, self.kernel_width, activation=self.activation_function),
#                                     MaxPooling1D(2),
#                                     Dropout(self.rate_drop_cnn),
#                                     Conv1D(256, self.kernel_width, activation=self.activation_function),
#                                     GlobalMaxPooling1D(),
#                                     Dropout(self.rate_drop_dense)
#                                     #Dense(2048, activation=self.activation_function, kernel_regularizer=l2(1e-3))
#                                     ])
#         
# =============================================================================

        # Connect CNN layer for First Sentence
        sequence_1_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = cnn_layer(embedded_sequences_1)

        # Connect CNN layer for Second Sentence
        sequence_2_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        x2 = cnn_layer(embedded_sequences_2)
        
        # TODO add lambda layer. See MNIST Siamese
        
        distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([x1, x2])
        
        #dense1 = Dense(1024, activation=self.activation_function)(distance)
        #dense2 = Dense(256, activation=self.activation_function)(dense1)
        
        # comment either one out
        #output = Dense(categories.shape[1], activation='sigmoid')(distance)
        output = Dense(1, activation='sigmoid')(distance)


        model = Model([sequence_1_input, sequence_2_input], output)


        # Merging two CNN
# =============================================================================
#         merged = concatenate([x1, x2])
#         
#         merged = BatchNormalization()(merged)
#         merged = Dropout(self.rate_drop_dense)(merged)
#         merged = Dense(self.number_dense_units, activation=self.activation_function)(merged)
#         merged = BatchNormalization()(merged)
#         merged = Dropout(self.rate_drop_dense)(merged)
#         preds = Dense(categories.shape[1], activation='sigmoid')(merged)
# =============================================================================

        #model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
        
        # comment either one out
        #model.compile(loss=self.loss_function, optimizer='nadam', metrics=['accuracy'])
        rms = RMSprop(lr=0.0001)
        model.compile(loss=self.contrastive_loss, optimizer=rms, metrics=['accuracy'])

        
        # print model
        model.summary()
        cnn_layer.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        STAMP = 'cnn_%d_%d_%.2f_%.2f' % (self.kernel_width, self.number_dense_units, self.rate_drop_cnn, self.rate_drop_dense)

        checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

        history = model.fit([train_data_x1, train_data_x2], train_labels,
                  validation_data=([val_data_x1, val_data_x2], val_labels),
                  epochs=8, batch_size=64, shuffle=True, verbose=1,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])
        
        plt.plot(history.history['loss'])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
        plt.plot(history.history['acc'])
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.show()

        return model

# TODO this method is not called yet
    def update_model(self, saved_model_path, new_sentences_pair, is_similar, embedding_meta_data):
        """
        Update trained siamese model for given new sentences pairs 
            Steps Involved:
                1. Pass the each from sentences from new_sentences_pair to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encodes and passed to dense layer.
                3. Pass the  dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            model_path (str): model path of already trained siamese model
            new_sentences_pair (list): list of tuple of new sentences pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix
        Returns:
            return (best_model_path):  path of best model
        """
        tokenizer = embedding_meta_data['tokenizer']
        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, new_sentences_pair,
                                                                               is_similar, self.max_sequence_length,
                                                                               self.validation_split_ratio)
        model = load_model(saved_model_path)
        model_file_name = saved_model_path.split('/')[-1]
        new_model_checkpoint_path  = saved_model_path.split('/')[:-2] + str(int(time.time())) + '/' 

        new_model_path = new_model_checkpoint_path + model_file_name
        model_checkpoint = ModelCheckpoint(new_model_checkpoint_path + model_file_name,
                                           save_best_only=True, save_weights_only=False)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        tensorboard = TensorBoard(log_dir=new_model_checkpoint_path + "logs/{}".format(time.time()))

        model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
                  epochs=50, batch_size=3, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return new_model_path