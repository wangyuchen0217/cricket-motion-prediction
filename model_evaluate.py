'''
This code will evaluate models from prediction results. 
Please manually input the parameters to "organize the time-slide window", "choose the output mode" and "set the model". 
Plots for different test objects will be stored at /Evaluation/Results/, and the evaluation csv. data will be stored at /Evaluation/.
Naming convention is "model_windowsize_timestep_(cricketnumber_)outcontent".
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import matplotlib.pyplot as plt
from window_generator import *
from model_set import *
from model_predict import *

if __name__ == '__main__':

    fold_path = os.getcwd() 
    ###### organize the time-slide window ######
    input_pattern = "pattern1" # ["pattern1(All)","pattern2(ThC+2FTi hind leg)","pattern3(THC)"]
    window_size = 100
    time_step = 20
    ###### choose the output mode ###### 
    out_mod = "mul" # ["sgl(single-step)","mul(multi-step)"]
    out_content = "Vel" # ["Vel","Direction"]
    test_trails = ["c16","c17","c18","c19","c20","c21"]
    ###### set the model ######
    model_type = "trans" # ["lstm","hlstm","arx","trans"]

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

    # load the model
    model_path = fold_path + "/Model/" + model_type + "_" + str(window_size) + "_" + str(time_step) + "_" + out_content + "_" + input_pattern + ".h5" 
    if model_type == 'arx':
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
        model = ARx(units=100, out_steps=time_step, input_num=input_num, output_num=output_num) 
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
        model.build(input_shape =(None, window_size,input_num))
        model.load_weights(model_path)
    elif model_type == 'trans':
        model = torch.load(model_path)
    else:
        model = tf.keras.models.load_model(model_path)

    # evaluation
    names = locals()
    if out_content == "Direction":
        results = np.zeros(5)
        for i in range(6):
            cricket_number = test_trails[i]
            names["results_" + cricket_number] = direction_results_visualization(model_type,
                                                                                window_size,
                                                                                time_step,
                                                                                cricket_number,
                                                                                input_pattern,
                                                                                fold_path)
            results = np.vstack((results, eval("results_" + cricket_number)))
        results = results[1:, :]
        evaluation = pd.DataFrame(data=results, columns=['RMSE(Direction_x)', 'P(Direction_x)',
                                                        'RMSE(Direction_y)', 'P(Direction_y)',
                                                        'R2(Direction)'])
    elif out_content == "Vel":
        results = np.zeros(7)
        for i in range(6):
            cricket_number = test_trails[i]
            names["results_" + cricket_number] = vel_results_visualization(model_type,
                                                                        window_size,
                                                                        time_step,
                                                                        cricket_number,
                                                                        input_pattern,
                                                                        fold_path)
            results = np.vstack((results, eval("results_" + cricket_number)))
        results = results[1:, :]
        evaluation = pd.DataFrame(data=results, columns=['RMSE(Vel)', 'P(Vel)',
                                                        'RMSE(Vel_x)', 'P(Vel_x)',
                                                        'RMSE(Vel_y)', 'P(Vel_y)',
                                                        'R2(Vel)'])
    means = evaluation.mean()
    stds = evaluation.std()
    evaluation.loc[6] = means
    evaluation.loc[7] = stds
    evaluation.index = ["c16", "c17", "c18", "c19", "c20", "c21", "mean", "std"]

    # save the evaluation results
    evaluation_path = fold_path + "/Evaluation/" + model_type + "_" + str(window_size) + "_" + str(time_step) + "_" + out_content + "_" + input_pattern + ".csv"
    evaluation.to_csv(path_or_buf=evaluation_path, header=True, index=True)
