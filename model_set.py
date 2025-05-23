'''
This code provides fuctions for building models with neural networks.
'''

import tensorflow as tf
from tensorflow import keras
import torch
from torch import nn
import math

'''lstm'''
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

'''hlstm'''
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
                
'''transformer'''
class PositionalEncoding(nn.Module):
# input and output shape: (sequence_length, batch_size, feature_size)
    def __init__(self, d_model, max_len=500):
    # max_len: the maximum length of the input sequence
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
    def __init__(self,feature_size,target_size,nhead,num_layers,dropout): 
    # feature_size: the dimension of features (Must be an integer multiple of head)
    # num_layers: the layers of Encoder_layer
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, 
                                                                            dropout=dropout, batch_first=True) 
        # the data shape is (batch_first, seq_len, feature_size) from the data loader, turn on batch_first
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,target_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1   
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != src.size(1):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src.transpose(0, 1)).transpose(0,1)
        # (batch_size, sequence_length, feature_size) -> (sequence_length, batch_size, feature_size) for pos_encoder
        # (sequence_length, batch_size, feature_size) -> (batch_size, sequence_length, feature_size) for transformer_encoder
        output = self.transformer_encoder(src,self.src_mask)
        output = self.decoder(output) 
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
def train_one_epoch(model, training_loader, loss_fn, optimizer, device):
    running_loss = 0.
    last_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # Add padding token to the output labels
        # len(labels[0,0,:])-> the number of output features
        padding_token =  torch.zeros(len(labels[0,0,:])) # choose any value for the padding token
        max_input_seq_len = inputs.shape[1] 
        padded_labels = torch.nn.utils.rnn.pad_sequence([torch.cat([seq, padding_token.repeat(max_input_seq_len - seq.shape[0], 1)]) for seq in labels], 
                                                                                                                    batch_first=True)
        # Create a binary mask for the padded tokens
        mask = torch.tensor(padded_labels != padding_token, dtype=torch.float32)
        # Slice the padded_labels to match the expected output shape
        padded_labels = padded_labels[:, :max_input_seq_len, :]

        # Verify the shapes match
        assert padded_labels.shape == (inputs.shape[0] , max_input_seq_len, labels.shape[2])
        assert mask.shape == (inputs.shape[0] , max_input_seq_len, labels.shape[2])
        #print(mask.shape)
        #print(padded_labels.shape)

        # Move data to the device
        inputs = inputs.to(device)
        padded_labels = padded_labels.to(device)
        mask = mask.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        # Mask out the padded tokens
        loss = loss_fn(outputs * mask, padded_labels * mask)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss

def train(EPOCHS, model, training_loader, loss_fn, optimizer, device):
    epoch_number = 0
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_one_epoch(model, training_loader, loss_fn, optimizer, device)
        epoch_number += 1