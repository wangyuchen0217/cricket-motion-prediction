import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from window_generator import *
from model_predict import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
import optuna
import warnings
from model_set import *

def get_R2(y_pred,y):
    R2_value = 1 - np.sum((y_pred - y)**2)/np.sum(y**2)
    return R2_value

def objective(trial):
    # set the model type 
    MODEL_TYPE = "trans" # ["lstm", "hlstm", "arx","trans"]

    #############################################################################
    # introduce the dataset
    fold_path = os.getcwd() 
    subjects = 33
    INPUT_NUM = 12
    OUTPUT_NUM = 3
    WINDOW_SIZE = 100
    if MODEL_TYPE == "lstm" or "hlstm":
        OUT_MOD = "sgl"
        TIME_STEP = 1
    elif MODEL_TYPE == "arx" or "trans":
        OUT_MOD = "mul"
        TIME_STEP = 10
    test_trails = ["c16", "c17","c18","c19","c20","c21"]
    ds_train = np.zeros(shape=(1,INPUT_NUM+OUTPUT_NUM))
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
            dataset_type =  trail_details[f"T{subject_number}"]["dataset_type"]
        if dataset_type == "train":
            ds_train_temp = get_dataset(subject_number, "Vel", "pattern1", fold_path)
            ds_train = np.vstack((ds_train, ds_train_temp))
        elif dataset_type == "test":
            names = locals()
            for i in range(6):
                if cricket_number == test_trails[i]:
                    names["ds_test_" + cricket_number ] = get_dataset(subject_number, "Vel", "pattern1", fold_path)
    ds_train = ds_train[1:,:]
    # normalization
    ds = ds_train
    for i in range(6):
        cricket_number = test_trails[i]
        ds = np.vstack((ds, eval("ds_test_" + cricket_number)))
    _, X_scaled, _, y_scaled = dataset_scaled(ds, "Vel")
    X_train_scaled, y_train_scaled = X_scaled[0:len(ds_train),:], y_scaled[0:len(ds_train),:]
    print("X_train_scaled.shape: (%2d, %2d)" %(X_train_scaled.shape[0], X_train_scaled.shape[1]))
    print("y_train_scaled.shape: (%2d, %2d)" %(y_train_scaled.shape[0], y_train_scaled.shape[1]))
    print("")
    # create input and output sequence of train & test set
    X_train, y_train = create_inout_sequences(X_train_scaled, y_train_scaled, WINDOW_SIZE, TIME_STEP, OUT_MOD, MODEL_TYPE) 
    print("X_train.shape: (%2d, %2d, %2d)" %(X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    print("y_train.shape: (%2d, %2d, %2d)" %(y_train.shape[0], y_train.shape[1], y_train.shape[2]))

    #############################################################################
    # introduce the model
    keras.backend.clear_session() 
    EPOCHS = 100     
    DROPOUT_RATIO = 0.5
    BATCHSIZE = 256

    # set which hyper parameters to search
    # for common
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    # for lstm, hlstm, arx
    num_hidden = trial.suggest_int("n_units{}".format(i), 4, 128, log=True)
    weight_decay_lstm = trial.suggest_float("weight_decay_lstm", 1e-10, 1e-3, log=True)
    weight_decay_dense = trial.suggest_float("weight_decay_dense", 1e-10, 1e-3, log=True)
    # for trans
    layer_num = trial.suggest_int("layer_num", 2, 12, log=True)
    hidden_size = trial.suggest_int("n_units{}".format(i), 4, 128, log=True)
    num_heads = trial.suggest_categorical("n_heads{}".format(i), [2, 3, 4, 6])
    dropout_ratio = trial.suggest_float("dropout_ratio", 0.1, 0.5, log=True)
    
    #############################################################################
    if MODEL_TYPE == "lstm":
        # set LSTM model
        model = keras.Sequential()
        model.add(keras.layers.LSTM(num_hidden, 
                                    return_sequences=False, 
                                    input_shape=(WINDOW_SIZE, INPUT_NUM),
                                    name="input_layer", 
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay_lstm)))
        model.add(keras.layers.Dropout(DROPOUT_RATIO,name="dropout_layer"))
        model.add(keras.layers.Dense(TIME_STEP*OUTPUT_NUM, 
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay_dense), 
                                    name="output_layer"))
        model.add(keras.layers.Reshape([TIME_STEP,OUTPUT_NUM]))
        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=0.5))
    
    #############################################################################
    elif MODEL_TYPE == "hlstm":
        # set HRNN model
        model = keras.Sequential()
        model.add(keras.layers.Dense(num_hidden, 
                                    input_shape=(WINDOW_SIZE,INPUT_NUM), 
                                    activation = "tanh",
                                    name="input_layer"))
        model.add(keras.layers.LSTM(num_hidden, 
                                    return_sequences=False, 
                                    name="dynamic_layer", 
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay_lstm)))
        model.add(keras.layers.Dropout(DROPOUT_RATIO,name="dropout_layer"))
        model.add(keras.layers.Dense(TIME_STEP*OUTPUT_NUM, 
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay_dense), 
                                    name="output_layer"))
        model.add(keras.layers.Reshape([TIME_STEP,OUTPUT_NUM]))
        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=0.5))
    
    #############################################################################
    elif MODEL_TYPE == "arx":
        # set ARx model
        class ARx(tf.keras.Model):
            def __init__(self, num_hidden, TIME_STEP):
                super().__init__()
                self.TIME_STEP = TIME_STEP
                self.num_hidden = num_hidden
                self.lstm_cell = tf.keras.layers.LSTMCell(num_hidden, 
                                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay_lstm))
                # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
                self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
                self.dropout = tf.keras.layers.Dropout(rate=0.5)
                self.dense = tf.keras.layers.Dense(INPUT_NUM+OUTPUT_NUM,
                                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay_dense))

        model = ARx(num_hidden=num_hidden, TIME_STEP=TIME_STEP) 

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
            predictions.append(prediction[:,:OUTPUT_NUM])

            # Run the rest of the prediction steps.
            for n in range(1, self.TIME_STEP):
                # Use the last prediction as input.
                x = prediction[:,OUTPUT_NUM:]
                # Execute one lstm step.
                x, state = self.lstm_cell(x, states=state, training=training)
                # Convert the lstm output to a prediction.
                prediction = self.dense(x) 
                # Add the prediction to the output.
                predictions.append(prediction[:,:OUTPUT_NUM])

            # predictions.shape => (time, batch, features)
            predictions = tf.stack(predictions)
            # predictions.shape => (batch, time, features)
            predictions = tf.transpose(predictions, [1, 0, 2])
            return predictions

        ARx.call = call
        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=0.5))

    #############################################################################    
    elif MODEL_TYPE == "trans":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransAm(feature_size=INPUT_NUM,
                                            target_size=OUTPUT_NUM,
                                            nhead=num_heads,
                                            hidden_size=hidden_size,
                                            num_layers=layer_num,
                                            dropout=dropout_ratio).to(device)
        training_loader_tensor = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        training_loader = torch.utils.data.DataLoader(training_loader_tensor, 
                                                                                                        batch_size=32,
                                                                                                        shuffle=True, 
                                                                                                        num_workers=16, 
                                                                                                        pin_memory=True, 
                                                                                                        persistent_workers=True,
                                                                                                        drop_last=True)
        loss =torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #############################################################################
    # cross-validation
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X_train_scaled):
        print("Train index:", train_index[:10], len(train_index))
        print("Test index:",test_index[:10], len(test_index))
        print('')
    scores = []
    for train_index, test_index in kf.split(X_train):
        X_train_i, X_test_i, y_train_i, y_test_i = (X_train[train_index, :], 
                                                    X_train[test_index, :], 
                                                    y_train[train_index, :], 
                                                    y_train[test_index, :])
        if MODEL_TYPE == "lstm" or MODEL_TYPE == "hlstm":
            model.fit(X_train_i,
                    y_train_i, 
                    shuffle=True,
                    batch_size=BATCHSIZE,
                    epochs=EPOCHS,
                    verbose=False)
            y_pred_i = model.predict(X_test_i)
            pp = y_pred_i.flatten('F')
            ll = y_test_i.flatten('F') 
        elif MODEL_TYPE == "arx":
            model.fit(X_train_i,
                    y_train_i[:,:,:OUTPUT_NUM], # ARx: y_train_i should add [:,:,:output_num] behind
                    shuffle=True,
                    batch_size=BATCHSIZE,
                    epochs=EPOCHS,
                    verbose=False)
            y_pred_i = model.predict(X_test_i)
            pp = y_pred_i.flatten('F')
            ll = y_test_i[:,:,:OUTPUT_NUM].flatten('F') # ARx: y_test_i should add [:,:,:output_num] behind
        elif MODEL_TYPE == "trans":
            train(EPOCHS=EPOCHS, model=model, 
                    training_loader=training_loader, 
                    loss_fn=loss, optimizer=optimizer, 
                    device=device)
            y_pred_i = get_prediction_from_transformer(X_test_i, y_test_i, 
                                                        model, TIME_STEP, 
                                                        WINDOW_SIZE, OUTPUT_NUM, device)
            pp = y_pred_i.flatten('F')
            ll = y_test_i.flatten('F')
        score = get_R2(pp,ll)
        scores.append(score)
        accuracy = np.mean(scores)

    return accuracy

if __name__ == "__main__":
    warnings.warn(
        "Recent Keras release (2.4.0) simply redirects all APIs "
        "in the standalone keras package to point to tf.keras. "
        "There is now only one Keras: tf.keras. "
        "There may be some breaking changes for some workflows by upgrading to keras 2.4.0. "
        "Test before upgrading. "
        "REF:https://github.com/keras-team/keras/releases/tag/2.4.0"
    )
    study = optuna.create_study(storage='sqlite:///db_trans.sqlite3', study_name='trans', direction="maximize")
    #study = optuna.study.load_study(study_name='arx',storage='sqlite:///arx.sqlite3')
    study.optimize(objective, n_trials=50, timeout=None)

    #fig = optuna.visualization.plot_intermediate_values(study)
    #fig.show()

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

#optuna.visualization.plot_optimization_history(study)
#optuna.visualization.plot_param_importances(study)
#optuna.visualization.plot_parallel_coordinate(study)