from keras.layers import Dense, LSTM, merge, Input
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Model
from keras.layers.core import  Dropout

import numpy as np
max_sequence_size = 16
pos_vec_dimension = 38
classes_num = 14

# POS Tags Features
LSTM_POS_layer = LSTM(50, activation='tanh', dropout_W = 0.2, dropout_U = 0.2)
lstm_pos_input_1 = Input(shape=(max_sequence_size, pos_vec_dimension))
lstm_pos_out_1 = LSTM_POS_layer(lstm_pos_input_1)

LSTM_POS_layer2 = LSTM(50, activation='tanh', dropout_W = 0.2, dropout_U = 0.2)
lstm_pos_input_2 = Input(shape=(max_sequence_size, pos_vec_dimension))
lstm_pos_out_2 = LSTM_POS_layer2(lstm_pos_input_2)

#word
LSTM_word_1 = LSTM(100, activation='tanh', dropout_W = 0.25, dropout_U = 0.25)
lstm_word_input_1 = Input(shape=(max_sequence_size, 300))
lstm_word_out_1 = LSTM_word_1(lstm_word_input_1)

LSTM_word_2 = LSTM(100, activation='tanh', dropout_W = 0.25, dropout_U = 0.25)
lstm_word_input_2 = Input(shape=(max_sequence_size, 300))
lstm_word_out_2 = LSTM_word_2(lstm_word_input_2)

#dependency
LSTM_dep_1 = LSTM(50, activation='tanh', dropout_W = 0.2, dropout_U = 0.2)
lstm_dep_input_1 = Input(shape=(12, 77))
lstm_dep_out_1 = LSTM_dep_1(lstm_dep_input_1)

LSTM_dep_2 = LSTM(50, activation='tanh', dropout_W = 0.2, dropout_U = 0.2)
lstm_dep_input_2 = Input(shape=(12, 77))
lstm_dep_out_2 = LSTM_dep_2(lstm_dep_input_2)

'''
Dense_token = Dense(100)
Dense_token_input = Input(shape=(600,))
Dense_token_out = Dense_token(Dense_token_input)

Dense_pos = Dense(100)
Dense_pos_input = Input(shape=(56,))
Dense_pos_out = Dense_pos(Dense_pos_input)
'''

merged_feature_vectors \
    = Dense(50, activation='sigmoid')(Dropout(0.3)(merge([lstm_pos_out_1, lstm_pos_out_2, lstm_word_out_1, lstm_word_out_2, lstm_dep_out_1, lstm_dep_out_2
             ], mode='concat', concat_axis=-1)))

predictions = Dense(classes_num, activation='softmax')(merged_feature_vectors)
#my_model = Model(input=[lstm_pos_input_1, lstm_word_vec_input_1, lstm_dep_rel_input_1], output=predictions)
my_model = Model(input=[lstm_pos_input_1, lstm_pos_input_2, lstm_word_input_1, lstm_word_input_2, lstm_dep_input_1,
                        lstm_dep_input_2], output=predictions)
print my_model.summary()

