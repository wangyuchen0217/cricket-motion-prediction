'''
This code will split train sets and test sets from the dataset, normalize all data, train the model and make predictions. 
Please manually input the parameters to "organize the time-slide window", "choose the output mode" and "set the model". 
The trained models will be stored at /Model/, and the prediction csv. data will be stored at /Evaluation/Results/.
Naming convention is "model_windowsize_timestep_(cricketnumber_)outcontent".
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import numpy as np
import torch
from window_generator import *
from model_set import *
from model_predict import *
# from torchsummary import summary

if __name__ == '__main__':
    
    fold_path = os.getcwd() 
    ###### organize the input and output sequence ######
    input_pattern = "pattern1" # ["pattern1(All)","pattern2(ThC+2FTi hind leg)","pattern3(THC)"]
    window_size = 100
    time_step = 1
    ###### choose the test mode ###### 
    out_mod = "sgl" # ["sgl(single-step)","mul(multi-step)"]
    out_content = "Vel" # ["Vel","Direction"]
    test_trails = ["c16", "c17","c18","c19","c20","c21"]
    ###### set the model ######
    model_type = "trans" # ["lstm","hlstm","arx","trans"]
    node_number = 100 # ["lstm:83","hlstm:124","arx:100"]
    dropout_ratio = 0.5
    batch_size = 256
    epochs = 100
    loss = "mse"
    learning_rate=3.97e-4 # ["lstm:1.06e-3","hlstm:3.97e-4","arx:100"]
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=0.5)

    # generate the input_num of the sequence
    if input_pattern == "pattern1":
        input_num = 12
    elif input_pattern == "pattern2":
        input_num = 8
    elif input_pattern == "pattern3":
        input_num = 6
    # generate the output_num of the sequence
    if out_content == "Vel":
        output_num = 3 
    elif out_content == "Direction":
        output_num =2

    # input the total number of subjects
    subjects = 33
    if out_content == "Direction":
        ds_train = np.zeros(shape=(1,input_num+output_num))
    elif out_content == "Vel":
        ds_train = np.zeros(shape=(1,input_num+output_num))
    for i in range(subjects):
        i = i + 1
        if i < 10:
            subject_number = "0" + str(i)
        else:
            subject_number = str(i)
        # get dataset for train and test seperately
        # 1 test set for each cricket (c16, c17, c18, c19, c20, c21) 
        with open("/home/yuchen/Crickets_Walking_Motion_Prediction/trail_details.json", "r") as f:  
            trail_details = json.load(f)
            cricket_number =  trail_details[f"T{subject_number}"]["cricket_number"]
            video_number = trail_details[f"T{subject_number}"]["video_number"]
            dataset_type =  trail_details[f"T{subject_number}"]["dataset_type"]
        if dataset_type == "train":
            ds_train_temp = get_dataset(subject_number, out_content, input_pattern, fold_path)
            ds_train = np.vstack((ds_train, ds_train_temp))
        elif dataset_type == "test":
            names = locals()
            for i in range(6):
                if cricket_number == test_trails[i]:
                    names["ds_test_" + cricket_number ] = get_dataset(subject_number, out_content, input_pattern, fold_path)
    ds_train = ds_train[1:,:]
    # normalization
    ds = ds_train
    for i in range(6):
        cricket_number = test_trails[i]
        ds = np.vstack((ds, eval("ds_test_" + cricket_number)))
    X_scaler, X_scaled, y_scaler, y_scaled = dataset_scaled(ds, out_content)
    X_train_scaled, y_train_scaled = X_scaled[0:len(ds_train),:], y_scaled[0:len(ds_train),:]
    if out_content == "Direction":
        label_num = 2
    elif out_content == "Vel":
        label_num =3
    # get test sets scaled seperately
    names = locals()
    for i in range(6):
        cricket_number = test_trails[i]
        names["X_test_scaled_" + cricket_number ] = X_scaler.transform(eval("ds_test_" + cricket_number)[:,:-label_num])
        names["y_test_scaled_" + cricket_number ] = y_scaler.transform(eval("ds_test_" + cricket_number)[:,-label_num:])
    print("X_train_scaled.shape: (%2d, %2d)" %(X_train_scaled.shape[0], X_train_scaled.shape[1]))
    print("y_train_scaled.shape: (%2d, %2d)" %(y_train_scaled.shape[0], y_train_scaled.shape[1]))
    print("")
    # create input and output sequence of train & test set
    X_train, y_train = create_inout_sequences(X_train_scaled, y_train_scaled, window_size, time_step, out_mod, model_type) 
    names = locals()
    for i in range(6):
        cricket_number = test_trails[i]
        names["X_test_" + cricket_number ], names["y_test_" + cricket_number ] = create_inout_sequences(eval("X_test_scaled_" + cricket_number), 
                                                                                                                                                                                                                        eval("y_test_scaled_" + cricket_number), 
                                                                                                                                                                                                                        window_size, 
                                                                                                                                                                                                                        time_step, 
                                                                                                                                                                                                                        out_mod,
                                                                                                                                                                                                                        model_type)
    print("X_train.shape: (%2d, %2d, %2d)" %(X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    print("y_train.shape: (%2d, %2d, %2d)" %(y_train.shape[0], y_train.shape[1], y_train.shape[2]))

    # train the model
    if model_type == "lstm":
        model = create_lstm_model(node_number, 
                                                                    dropout_ratio,
                                                                    window_size, 
                                                                    input_num,  
                                                                    output_num,
                                                                    time_step)
        model.compile(loss=loss, optimizer=optimizer)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    elif model_type == "hlstm":
        model = create_hlstm_model(node_number, 
                                                                        dropout_ratio,
                                                                        window_size, 
                                                                        input_num,  
                                                                        output_num,
                                                                        time_step)
        model.compile(loss=loss, optimizer=optimizer)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    elif model_type == "arx":
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
        model = ARx(units=100, out_steps=10, input_num=12, output_num=3)    
        
        def warmup(self, inputs):
            # inputs.shape => (batch, time, features)
            # x.shape => (batch, lstm_units)
            x, *state = self.lstm_rnn(inputs)
            # predictions.shape => (batch, features)
            prediction = self.dense(x)
            return prediction, state
        ARx.warmup = warmup
    
        def call(self, inputs, training=None):
            # Use a TensorArray to capture dynamically unrolled outputs.
            predictions = []
            # Initialize the LSTM state.
            prediction, state = self.warmup(inputs)
            # Insert the first prediction.
            predictions.append(prediction[:,:output_num])
            # Run the rest of the prediction steps.
            for n in range(1, self.out_steps):
                # Use the last prediction as input.
                x = prediction[:,output_num:]
                # Execute one lstm step.
                x, state = self.lstm_cell(x, states=state, training=training)
                # Convert the lstm output to a prediction.
                prediction = self.dense(x) 
                # Add the prediction to the output.
                predictions.append(prediction[:,:output_num])
            # predictions.shape => (time, batch, features)
            predictions = tf.stack(predictions)
            # predictions.shape => (batch, time, features)
            predictions = tf.transpose(predictions, [1, 0, 2])
            return predictions
        ARx.call = call
        
        model.compile(loss=loss, optimizer=optimizer)
        history = model.fit(X_train, y_train[:,:,:output_num], epochs=epochs, batch_size=batch_size)

    elif model_type == "trans":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransAm(feature_size=input_num,
                                            target_size=output_num,
                                            nhead=4,
                                            num_layers=7,
                                            dropout=0.1).to(device)
        training_loader_tensor = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        training_loader = torch.utils.data.DataLoader(training_loader_tensor, 
                                                                                                        batch_size=32,
                                                                                                        shuffle=True, 
                                                                                                        num_workers=16, 
                                                                                                        pin_memory=True, 
                                                                                                        persistent_workers=True,
                                                                                                        drop_last=True)
        loss =torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train(EPOCHS=epochs, model=model, 
                    training_loader=training_loader, 
                    loss_fn=loss, optimizer=optimizer, 
                    device=device)
        

    # save the model
    model_path = fold_path + "/Model/" + model_type + "_" + str(window_size) + "_" + str(time_step) + "_" + out_content + "_" + input_pattern + ".h5" 
    if model_type == "arx":
        model.save_weights(model_path)
    elif model_type == "trans":
        torch.save(model, model_path)
    else:
        model.save(model_path) 

    # model_path = fold_path + "/Model/" + model_type + "_" + str(window_size) + "_" + str(time_step) + "_" + out_content + "_" + input_pattern + ".h5" 
    # model = torch.load(model_path)

    # get results
    for i in range (6):
        cricket_number = test_trails[i]
        X_test = eval("X_test_" + cricket_number)
        y_test_scaled = eval("y_test_scaled_" + cricket_number)
        get_results(X_test, y_test_scaled, out_mod, model, y_scaler, model_type, 
                                output_num, window_size, time_step, cricket_number, out_content, input_pattern, fold_path, device)