'''
This code provides functions for data normalization 
and create input and output sequences from the dataset.
'''

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

def get_dataset(subject:str, out_content, input_pattern, fold_path):
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
    if out_content == 'Vel':
        dataset_path = fold_path + '/Dataset/' + cricket_number + '/' + video_number + '_Dataset_Absolute.csv'
        if input_pattern == "pattern1":
            dataset = pd.read_csv(dataset_path, header=0, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18])
        elif input_pattern == "pattern2":
            dataset = pd.read_csv(dataset_path, header=0, usecols=[1, 2, 3, 4, 5, 6, 9, 12, 16, 17, 18])
        elif input_pattern == "pattern3":
            dataset = pd.read_csv(dataset_path, header=0, usecols=[1, 2, 3, 4, 5, 6, 16, 17, 18])
        dataset = np.array(dataset)
    if out_content == 'Direction':
        dataset_path = fold_path + '/Dataset/' + cricket_number + '/' + video_number + '_Dataset_Relative.csv'  
        if input_pattern == "pattern1":
            dataset = pd.read_csv(dataset_path, header=0, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15])
        elif input_pattern == "pattern2":
            dataset = pd.read_csv(dataset_path, header=0, usecols=[1, 2, 3, 4, 5, 6, 9, 12, 14, 15])
        elif input_pattern == "pattern3":
            dataset = pd.read_csv(dataset_path, header=0, usecols=[1, 2, 3, 4, 5, 6, 14, 15])
        dataset = np.array(dataset)
    return dataset

def dataset_scaled(dataset, out_content):
    # normalization
    if out_content == 'Direction':
        X = dataset[:, :-2]
        y = dataset[:, -2:]
    if out_content == 'Vel':
        X = dataset[:, :-3]
        y = dataset[:, -3:]
    X_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(y)
    # normalize X,y individually
    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(y)
    # de-normalization
    # predict = scaler.inverse_transform(all_data_scaled)
    return X_scaler, X_scaled, y_scaler, y_scaled

def create_inout_sequences(X, y, window_size, time_step, out_mod, model_type):
    input = []
    output = []
    lenth = len(X)
    width = len(X[0])
    # lenth-(window_size+future_step)+1 = lenth-window_size-(future_step-1)
    for i in range(lenth - window_size - time_step+1):
        if out_mod == "sgl":
            if model_type == ("lstm" or "hlstm"):
                feature = X[i:i+window_size, :]
                label = y[i+window_size, :].reshape(1,-1)
            elif model_type == "arx":
                feature = X[i:i+window_size, :]
                label = np.hstack((y[i+window_size, :].reshape(1,-1),
                                                    X[i+window_size, :].reshape(1,-1)))
            elif model_type == "trans":
                feature = X[i:i+window_size,:]
                label = y[i+window_size, :].reshape(1,-1)
        elif out_mod == "mul":
            if model_type == ("lstm" or "hlstm"):
                feature = X[i:i+window_size, :]
                label = y[i+window_size:i+window_size+time_step, :] 
            elif model_type == "arx":
                feature = X[i:i+window_size, :]
                label = np.hstack((y[i+window_size:i+window_size+time_step, :],
                                                    X[i+window_size:i+window_size+time_step, :]))
            elif model_type == "trans":
                feature = X[i:i+window_size, :]
                label = y[i+window_size:i+window_size+time_step, :] 
        input.append(feature)
        output.append(label)
    input = np.array(input)
    output = np.array(output)
    return input, output
