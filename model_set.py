'''
This code provides fuctions for building models with neural networks.
'''

import tensorflow as tf
from tensorflow import keras
import torch
from torch import nn
from torch import Generator
import transformer.Constants
import transformer.Modules
import transformer.Layers
import transformer.SubLayers
import transformer.Models
import transformer.Translator
import transformer.Optim
import copy

from torchsummary import summary

def create_lstm_model(node_number, 
                                                dropout_ratio,
                                                window_size, 
                                                input_num,  
                                                output_num,
                                                time_step):
    keras.backend.clear_session() 
    model = keras.Sequential()
    model.add(keras.layers.LSTM(node_number, 
                                                                return_sequences=False, 
                                                                input_shape=(window_size, input_num),
                                                                name="input_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l2(l=8.49e-7)))
    model.add(keras.layers.Dropout(dropout_ratio,name="dropout_layer"))
    model.add(keras.layers.Dense(time_step*output_num, 
                                                                    kernel_regularizer=tf.keras.regularizers.l2(l=2.81e-6), 
                                                                    name="output_layer"))
    model.add(keras.layers.Reshape([time_step,output_num]))
    model.summary()
    return model

def create_hlstm_model(node_number, 
                                                    dropout_ratio,
                                                    window_size, 
                                                    input_num,  
                                                    output_num,
                                                    time_step):
    keras.backend.clear_session() 
    model = keras.Sequential()
    model.add(keras.layers.Dense(node_number, 
                                                                    input_shape=(window_size,input_num), 
                                                                    activation = "tanh",
                                                                    name="input_layer"))
    model.add(keras.layers.LSTM(node_number, 
                                                                return_sequences=False, 
                                                                name="dynamic_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l2(l=2.21e-7)))
    model.add(keras.layers.Dropout(dropout_ratio,name="dropout_layer"))
    model.add(keras.layers.Dense(time_step*output_num, 
                                                                    kernel_regularizer=tf.keras.regularizers.l2(l=3.75e-9), 
                                                                    name="output_layer"))
    model.add(keras.layers.Reshape([time_step,output_num]))
    model.summary()
    return model

class ARx(tf.keras.Model):
            def __init__(self, units, out_steps, input_num, output_num):
                super().__init__()
                self.out_steps = out_steps
                self.units = units
                self.lstm_cell = tf.keras.layers.LSTMCell(units, kernel_regularizer=tf.keras.regularizers.l2(l=1e-7))
                # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
                self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
                self.dropout = tf.keras.layers.Dropout(rate=0.5)
                self.dense = tf.keras.layers.Dense(input_num+output_num,
                                                kernel_regularizer=tf.keras.regularizers.l2(l=0.001))
                
'''
model = transformer.Models.Transformer(n_src_vocab=100, n_trg_vocab=10, src_pad_idx=None, trg_pad_idx=None,
            d_word_vec=12, d_model=12, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj')

for name,parameters in model.named_parameters():
    print(name,':',parameters.size())

# up to the nn.embedding, see transformer/Models.py
'''
