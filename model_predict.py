'''
This code provides functions for model predictions,
model predictions and result visualization.
Naming convention is "model_windowsize_timestep_(cricketnumber_)outcontent".
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import torch

def get_rmse(y_pred,y):
    size = y_pred.shape[0]
    rmse_value = np.sqrt(np.sum((y_pred - y)**2)/size)
    return rmse_value

def get_r2(y_pred,y):
    R2_value = 1 - np.sum((y_pred - y)**2)/np.sum(y**2)
    return R2_value

def get_p_value(y_pred,y):
    _,p_value = stats.ttest_ind(y_pred,y)
    return p_value

def get_prediction_from_estimation(input, model, output_num):
    m = len(input)
    for i in range(m):
        X = input[i, :, :][np.newaxis]
        if i == 0:
            Y_preds = model.predict(X, verbose=0)
        else:
            y_pred = model.predict(X, verbose=0)
            Y_preds = np.concatenate((Y_preds, y_pred))
    return np.reshape(Y_preds, (-1, output_num))

def get_prediction_from_sequence(input, output_scaled, 
                                                                            model, time_step, 
                                                                            window_size, output_num):
    m = len(input)
    n = len(output_scaled)
    remainder = (n-time_step-window_size) % time_step
    for i in range(0, m, time_step):
        X = input[i, :, :][np.newaxis]
        if i == 0:
            Y_preds = model.predict(X, verbose=0) 
        else:
            y_pred = model.predict(X, verbose=0)
            Y_preds = np.concatenate((Y_preds, y_pred),axis=1)
    if remainder != 0:
        remove = time_step - remainder
        Y_preds = Y_preds[:,:-remove,:]
    return np.reshape(Y_preds,(-1,output_num))

def get_prediction_from_recursive(input, output_scaled, 
                                                                            model, time_step, 
                                                                            window_size):
    predictions = model.call(input)
    m = len(input) 
    n = len(output_scaled)
    remainder = (n-time_step-window_size) % time_step
    Y_preds = []
    for i in range(0, m, time_step):
        if i == 0:
            Y_preds = predictions[i,:,:]
        else:
            y_pred = predictions[i,:,:]
            Y_preds = np.concatenate((Y_preds, y_pred),axis=0)
    if remainder != 0:
        remove = time_step - remainder
        Y_preds = Y_preds[:-remove,:]
    return Y_preds

def get_prediction_from_transformer(input, output_scaled, 
                                                                                model, time_step, 
                                                                                window_size, output_num, device):
    m = len(input) 
    n = len(output_scaled)
    remainder = (n-time_step-window_size) % time_step
    for i in range(0, m, time_step):
        X = input[i, :, :][np.newaxis]
        X = torch.from_numpy(X).float().to(device)
        if i == 0:
            Y_preds = model(X).detach().cpu().numpy()[:,:time_step,:] # [:,:10,:] for 10-step prediction
        else:
            y_pred = model(X).detach().cpu().numpy()[:,:time_step,:] # pred.shape: (1, 10, out_num)
            Y_preds = np.concatenate((Y_preds, y_pred),axis=1)
    if remainder != 0:
        remove = time_step - remainder
        Y_preds = Y_preds[:,:-remove,:]
    return np.reshape(Y_preds,(-1,output_num))

def get_results(X_test,
                                y_test_scaled,
                                out_mod, 
                                model,
                                y_scaler,
                                model_type, 
                                output_num,
                                window_size,
                                time_step,
                                cricket_number,
                                out_content, 
                                input_pattern,
                                fold_path,
                                device):
    pred_test_scaled = None
    label_test_scaled = None
    if out_mod == "sgl":
        if model_type == ("lstm" or "hlstm"):
            pred_test_scaled = get_prediction_from_estimation(X_test, model, output_num)
            label_test_scaled = y_test_scaled[window_size:,:]
        elif model_type == "trans":
            pass
    elif out_mod == "mul":
        if model_type == ("lstm" or "hlstm"):
            pred_test_scaled = get_prediction_from_sequence(X_test, y_test_scaled, 
                                                                                                                    model, time_step, 
                                                                                                                    window_size, output_num)
            label_test_scaled = y_test_scaled[window_size:-time_step,:]
        elif model_type == "arx":
            pred_test_scaled = get_prediction_from_recursive(X_test, y_test_scaled, 
                                                                                                                        model, time_step, 
                                                                                                                        window_size)
            label_test_scaled = y_test_scaled[window_size:-time_step,:]
        elif model_type == "trans":
            pred_test_scaled = get_prediction_from_transformer(X_test, y_test_scaled, 
                                                                                                                        model, time_step, 
                                                                                                                        window_size, output_num, device)
            label_test_scaled = y_test_scaled[window_size:-time_step,:]
    # de-normalization
    pred_test = y_scaler.inverse_transform(pred_test_scaled)
    label_test = y_scaler.inverse_transform(label_test_scaled)
    print("pred_test.shape: (%2d, %2d)" %(pred_test.shape[0],pred_test.shape[1]))
    print("label_test.shape: (%2d, %2d)" %(label_test.shape[0],label_test.shape[1]))
    #save the prediction results
    prediction_results = np.hstack((pred_test, label_test))
    if out_content == "Direction":
        df_prediction_results = pd.DataFrame(data=prediction_results, 
                                                                                        columns=["pred_direction_x", "pred_direction_y", "label_direction_x", "label_direction_y"])
    elif out_content == "Vel":
        df_prediction_results = pd.DataFrame(data=prediction_results, 
                                                                                        columns=["pred_vel", "pred_vel_x", "pred_vel_y", "label_vel", "label_vel_x", "label_vel_y"])
    results_path = fold_path + "/Evaluation/Results/" + model_type + "_" + str(window_size) + "_" + str(time_step) + "_" + cricket_number + "_" + out_content + "_" + input_pattern + ".csv"
    df_prediction_results.to_csv(path_or_buf=results_path, header=True, index=True)

def direction_results_visualization(model_type,
                                                                        window_size,
                                                                        time_step,
                                                                        cricket_number, 
                                                                        input_pattern,
                                                                        fold_path):
    results_path = fold_path + "/Evaluation/Results/" + model_type + "_" + str(window_size) + "_" + str(time_step) + "_" + cricket_number + "_Direction_" + input_pattern +  ".csv"
    results = pd.read_csv(results_path, header=0, usecols=[1, 2, 3, 4])
    results = np.array(results)
    
    pred_test = results[:,:2]
    label_test = results[:,2:]

    end = len(label_test) * 1/119.88
    t_test = np.arange(0, end, 1/119.88) 
    
    plt.figure(figsize=(12, 3))
    plt.plot(t_test,label_test[:,0],label='Original data',c='blue',linewidth=2)
    plt.plot(t_test,pred_test[:,0],label='Prediction',c='red',linestyle='--',linewidth=2)
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)
    plt.xlabel('Time t [s]',fontsize=14)
    plt.ylabel('Direction_x [vec]',fontsize=14)
    plt.title('Direction_x_'+cricket_number,fontsize=14)
    plt.savefig(fold_path + "/Evaluation/Results/" + model_type + "_" + str(window_size) + "_" + str(time_step) + "_" + cricket_number + "_Direction_x_" + input_pattern +  ".png",bbox_inches = 'tight')
    #plt.legend(fontsize=14)

    plt.figure(figsize=(12, 3))
    plt.plot(t_test,label_test[:,1],label='Original data',c='blue',linewidth=2)
    plt.plot(t_test,pred_test[:,1],label='Prediction',c='red',linestyle='--',linewidth=2)
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)
    plt.xlabel('Time t [s]',fontsize=14)
    plt.ylabel('Direction_y [vec]',fontsize=14)
    plt.title('Direction_y_'+cricket_number,fontsize=14)
    plt.savefig(fold_path + "/Evaluation/Results/" + model_type + "_" + str(window_size) + "_" + str(time_step) + "_" + cricket_number + "_Direction_y_" + input_pattern +  ".png",bbox_inches = 'tight')
    #plt.legend(fontsize=14)
    
    direction_x_rmse = get_rmse(pred_test[:,0],label_test[:,0])
    direction_y_rmse = get_rmse(pred_test[:,1],label_test[:,1])
    direction_r2 = get_r2(pred_test,label_test)
    direction_x_p = get_p_value(pred_test[:,0],label_test[:,0])
    direction_y_p = get_p_value(pred_test[:,1],label_test[:,1])
    results = np.array([direction_x_rmse, direction_x_p, direction_y_rmse, direction_y_p, direction_r2])
    return results

def vel_results_visualization(model_type,
                                                            window_size,
                                                            time_step,
                                                            cricket_number, 
                                                            input_pattern,
                                                            fold_path):
    results_path = fold_path + "/Evaluation/Results/" + model_type + "_" + str(window_size) + "_" + str(time_step) + "_" + cricket_number + "_Vel_" + input_pattern +  ".csv"
    results = pd.read_csv(results_path, header=0, usecols=[1, 2, 3, 4, 5, 6])
    results = np.array(results)
    
    pred_test = results[:,:3]
    label_test = results[:,3:]

    end = len(label_test) * 1/119.88
    t_test = np.arange(0, end, 1/119.88) 

    plt.figure(figsize=(12, 3))
    plt.plot(t_test,label_test[:,0],label='Original data',c='blue',linewidth=2)
    plt.plot(t_test,pred_test[:,0],label='Prediction',c='red',linestyle='--',linewidth=2)
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)
    plt.xlabel('Time t [s]',fontsize=14)
    plt.ylabel('Vel [mm/s]',fontsize=14)
    plt.title('Vel_'+cricket_number,fontsize=14)
    plt.savefig(fold_path + "/Evaluation/Results/" + model_type + "_" + str(window_size) + "_" + str(time_step) + "_" + cricket_number + "_Vel_" + input_pattern +  ".png",bbox_inches = 'tight')
    #plt.legend(fontsize=14)
    
    plt.figure(figsize=(12, 3))
    plt.plot(t_test,label_test[:,1],label='Original data',c='blue',linewidth=2)
    plt.plot(t_test,pred_test[:,1],label='Prediction',c='red',linestyle='--',linewidth=2)
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)
    plt.xlabel('Time t [s]',fontsize=14)
    plt.ylabel('Vel_x [mm/s]',fontsize=14)
    plt.title('Vel_x_'+cricket_number,fontsize=14)
    plt.savefig(fold_path + "/Evaluation/Results/" + model_type + "_" + str(window_size) + "_" + str(time_step) + "_" + cricket_number + "_Vel_x_" + input_pattern +  ".png",bbox_inches = 'tight')
    #plt.legend(fontsize=14)

    plt.figure(figsize=(12, 3))
    plt.plot(t_test,label_test[:,2],label='Original data',c='blue',linewidth=2)
    plt.plot(t_test,pred_test[:,2],label='Prediction',c='red',linestyle='--',linewidth=2)
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)
    plt.xlabel('Time t [s]',fontsize=14)
    plt.ylabel('Vel_y [mm/s]',fontsize=14)
    plt.title('Vel_y_'+cricket_number,fontsize=14)
    plt.savefig(fold_path + "/Evaluation/Results/" + model_type + "_" + str(window_size) + "_" + str(time_step) + "_" + cricket_number + "_Vel_y_" + input_pattern +  ".png",bbox_inches = 'tight')
    #plt.legend(fontsize=14)
    
    vel_rmse = get_rmse(pred_test[:,0],label_test[:,0])
    vel_x_rmse = get_rmse(pred_test[:,1],label_test[:,1])
    vel_y_rmse = get_rmse(pred_test[:,2],label_test[:,2])
    vel_r2 = get_r2(pred_test,label_test)
    vel_p = get_p_value(pred_test[:,0],label_test[:,0])
    vel_x_p = get_p_value(pred_test[:,1],label_test[:,1])
    vel_y_p = get_p_value(pred_test[:,2],label_test[:,2])
    results = np.array([vel_rmse, vel_p, vel_x_rmse, vel_x_p, vel_y_rmse, vel_y_p, vel_r2])
    return results
