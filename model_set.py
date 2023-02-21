'''
This code provides fuctions for building models with neural networks.
'''

import tensorflow as tf
from tensorflow import keras
import torch
from torch import nn
#from torchsummary import summary

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
                
'''transformer'''
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
          

class TransAm(nn.Module):
    def __init__(self,feature_size=250,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
