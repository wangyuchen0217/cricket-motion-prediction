'''
This code provides fuctions for building models with neural networks.
'''

import tensorflow as tf
from tensorflow import keras

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