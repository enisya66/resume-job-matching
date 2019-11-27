# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:55:35 2019

@author: User
"""

# keras imports
from keras.layers import Dense, Input, LSTM, SpatialDropout1D, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Lambda, Flatten, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop, Adam
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
        # TODO use bias?
        # TODO multi filters (2,3,4) + concat
        cnn_layer = Sequential([Conv1D(200, self.kernel_width, activation=self.activation_function),
                                #Dropout(0.2),
                                GlobalMaxPooling1D(),
                                Dense(200, activation=self.activation_function),
                                Dropout(0.4)
                                #BatchNormalization(),
                                #MaxPooling1D(3),
                                #SpatialDropout1D(self.rate_drop_cnn),

                                #Conv1D(128, self.kernel_width, activation=self.activation_function),
                                #BatchNormalization(),
                                #MaxPooling1D(3),
                                #SpatialDropout1D(self.rate_drop_cnn),
                                #Conv1D(128, 3, activation=self.activation_function),
                                #GlobalMaxPooling1D(),
                                #MaxPooling1D(3),
                                #BatchNormalization(),
                                #Dropout(self.rate_drop_cnn),
                                #Reshape((-1, 128))
                                #Dense(256, activation=self.activation_function, kernel_regularizer=l2(1e-3)),
                                #Dropout(self.rate_drop_cnn)
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

        # Connect CNN layer for first sentence
        sequence_1_input = Input(shape=(self.max_sequence_length,))
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        #average_embedded_1 = Lambda(lambda x: K.mean(x, axis=1))(embedded_sequences_1)
        #average_embedded_1 = np.expand_dims(average_embedded_1, axis=-1)
        x1 = cnn_layer(embedded_sequences_1)

        # Connect CNN layer for second sentence
        sequence_2_input = Input(shape=(self.max_sequence_length,))
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        #average_embedded_2 = Lambda(lambda x: K.mean(x, axis=1))(embedded_sequences_2)
        #average_embedded_2 = np.expand_dims(average_embedded_2, axis=-1)
        x2 = cnn_layer(embedded_sequences_2)
           
        distance = Lambda(self.cosine_distance, output_shape=self.eucl_dist_output_shape)([x1, x2])
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
        model.compile(loss=self.loss_function, optimizer='adam', metrics=['accuracy'])
        #model.compile(loss=self.contrastive_loss, optimizer=rms)

        # print model
        model.summary()
        cnn_layer.summary()

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
                  epochs=30, batch_size=256, shuffle=True, verbose=1,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])
        
		# plot metrics graphs
        plt.plot(history.history['loss'], 'bo', label='Loss')
        plt.plot(history.history['val_loss'], 'b', label='Validation')
        plt.xlabel('epochs')
        plt.ylabel('loss')
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