import tensorflow as tf
from tensorflow import keras
import torch
from torch import nn
import numpy as np
import math
from torchsummary import summary

class PositionalEncoding(nn.Module):

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
    def __init__(self,feature_size,target_size,nhead, num_layers,dropout): 
    # feature_size: the dimension of features (Must be an integer multiple of head)
    # num_layers: the layers of Encoder_layer
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, 
                                                                                                                dropout=dropout, batch_first=True)
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
            # permute the dimensions to (sequence length, batch size, embedding dimension)
            mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
            # mask use the length of the input sequence, so use len(src[1]) instead of len(src)
            self.src_mask = mask

        print("input_src.shape:" + str(src.shape)) # (32, 110, 12)
        src = self.pos_encoder(src.transpose(0,1)).transpose(0,1)
        print("af_po_encode_src.shape:" + str(src.shape)) # (32, 110, 12)
        output = self.transformer_encoder(src,self.src_mask)
        print("encoder_output.shape:" + str(output.shape)) # (32, 110, 12)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

encoder_layer = nn.TransformerEncoderLayer(d_model=12, nhead=4)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

src = torch.rand(100, 110, 12) # 32: batch size, 110: sequence length, 12: feature size
target = torch.rand(100, 110, 3) # 3: target size

X_train = src.numpy()
y_train = target.numpy()
training_dataset = np.concatenate((X_train, y_train), axis=2)
training_loader_tensor = torch.from_numpy(training_dataset).float()
training_loader = torch.utils.data.DataLoader(training_loader_tensor, 
                                                                                                        batch_size=32,
                                                                                                        shuffle=True, 
                                                                                                        num_workers=16, 
                                                                                                        pin_memory=True, 
                                                                                                        persistent_workers=True,
                                                                                                        drop_last=True)
model = TransAm(feature_size=12,
                                            target_size=3,
                                            nhead=4,
                                            num_layers=1,
                                            dropout=0.1)
for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        # input, labels = data
        inputs = data[:, :, :-3]
        labels = data[:, :, -3:]

        out = model(inputs)
        print("out.shape:" + str(out.shape)) # (32, 110, 3)
        print("labels.shape:" + str(labels.shape)) # (32, 110, 3
        loss_fn =torch.nn.MSELoss()
        loss = loss_fn(out, labels)
