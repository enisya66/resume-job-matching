# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:21:05 2019

@author: User
"""

# keras imports
from keras.layers import Dense, Input, LSTM, SpatialDropout1D, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Lambda, Flatten, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard
from keras.models import Model, Sequential
from keras import backend as K

# std imports
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# TODO create train dev set w/o leaks
from EmbeddingUtils import create_train_dev_set


class SiameseMultiCNN:
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
        return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))
    
    def cosine_distance(self, vects):
        x, y = vects
        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)
        return -K.mean(x * y, axis=-1, keepdims=True)
    
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
        Trains Siamese network to find similarity between sentences in `sentences_pair`
        Args:
            sentences_pair (list): list of tuple of sentence pairs
            categories (list): target values (1-5)
			tokenizer (keras.Tokenizer): keras Tokenizer object containing word indexes
			embedding_matrix (np.array): matrix of word indexes and respective word vectors
            model_save_directory (str): working directory to save models
        Returns:
            model: trained keras model
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

        # TODO multi filters (2,3,4) + concat
        kernel_size = [3,4]
        cnn_layer = []
        inp = Input(shape=(self.max_sequence_length, 300, ))
        
        for i in kernel_size:
            #submodel = Sequential()
            conv = Conv1D(128, i, activation=self.activation_function)(inp)
            pool = MaxPooling1D(i)(conv)
            flat = Flatten()(pool)
            drop = Dropout(0.4)(flat)
            cnn_layer.append(drop)
        

        out = Concatenate()(cnn_layer)
        conv_model = Model(input=inp, output=out)
          

        # Connect CNN layer for first sentence
        sequence_1_input = Input(shape=(self.max_sequence_length,))
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = conv_model(embedded_sequences_1)

        # Connect CNN layer for second sentence
        sequence_2_input = Input(shape=(self.max_sequence_length,))
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        x2 = conv_model(embedded_sequences_2)
        
        #x1 = cnn_layer(sequence_1_input)
        #x2 = cnn_layer(sequence_2_input)
           
        distance = Lambda(self.manhattan_distance, output_shape=self.eucl_dist_output_shape)([x1, x2])
        # if dense layer after concat/merging
        #dense1 = Dense(1024, activation=self.activation_function)(distance)
        #dense2 = Dense(256, activation=self.activation_function)(dense1)
        
        # comment either one out
        #output = Dense(categories.shape[1], activation='softmax')(distance)
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
        rms = RMSprop()
        #adam = Adam(lr=0.0001)
        model.compile(loss=self.loss_function, optimizer=rms, metrics=['accuracy'])
        #model.compile(loss=self.contrastive_loss, optimizer=rms)

        # print model
        model.summary()
        conv_model.summary()

		# stops training when there's no improvement
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        STAMP = 'cnn_%d_%d_%.2f_%.2f' % (self.kernel_width, self.number_dense_units, self.rate_drop_cnn, self.rate_drop_dense)

        checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))
		
		# happy training
        history = model.fit([train_data_x1, train_data_x2], train_labels,
                  validation_data=([val_data_x1, val_data_x2], val_labels),
                  epochs=8, batch_size=64, shuffle=True, verbose=1,
                  callbacks=[model_checkpoint, tensorboard])
        
		# plot metrics graphs
        plt.plot(history.history['loss'], 'bo', label='Loss')
        plt.plot(history.history['val_loss'], 'b', label='Validation')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()

        return model