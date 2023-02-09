'''
This code will do preprocessing for all data including velocity, direction, and joint angle data. 
The skeleton analysis data is from /DataPreparation/Skeleton_data/, 
and velocity data measured by optical flow sensors is from /DataPreparation/Velocity_data/.
The generate preprocessed csv. data will be stored at /DataPreparation/Preprocessed_data/.
'''
import os
import json
import numpy as np
import pandas as pd
from pykalman import KalmanFilter 

def data_preprocess (subject:str, fold_path):
    save_vel_smooth(subject, fold_path)
    save_dlc_crop(subject, fold_path)

''' 
Velocity data preprocessing: 
Upsample velocity data from 50 Hz to 120 Hz, to synchronize with video data. Then smooth it using Kalman filtering.
'''
def vel_resample(subject:str, fold_path):
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
    vel_path = fold_path+'/DataPreparation/Velocity_data/'+cricket_number+'/'+video_number+'_Vel.csv'
    vel = pd.read_csv(vel_path, header=None, index_col=[0], usecols=[0,1,2,3,4])
     # change the original index formatt
    DatetimeIndex = pd.to_datetime(vel.index, unit='s',origin=pd.Timestamp('2022-5-27 00:00:00')) 
    vel.index = DatetimeIndex
    # upsample to 59.94 (60) fps 16683us
    # upsample to 119.88 (120) fps / 8342us
    vel_r = vel.resample('8342us').ffill() 
    # delete the 1st row since it is NAN due to the "ffill" method
    vel_r.drop(vel_r.index[[0]],inplace=True) 
    print("Video: %s, Resampled vel shape: (%2d, %2d)" %(video_number,
                                                         int(vel_r.shape[0]),
                                                         int(vel_r.shape[1])))
    return vel_r

def Kalman1D(observations,damping=1):
    # to return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.03
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state

def vel_smooth(vel_r):
    a = np.array(vel_r)
    # kalman filtering
    velocity = a[:,0]
    velocity_kal = Kalman1D(velocity,damping=1)
    velocity_x = a[:,1]
    velocity_x_kal = Kalman1D(velocity_x,damping=1)
    velocity_y = a[:,2]
    velocity_y_kal = Kalman1D(velocity_y,damping=1)
    ang_vel = a[:,3]
    ang_vel_kal = Kalman1D(ang_vel,damping=1)
    # hstack
    velocity_kal = velocity_kal.reshape(-1,1)
    velocity_x_kal = velocity_x_kal.reshape(-1,1)
    velocity_y_kal = velocity_y_kal.reshape(-1,1)
    ang_vel_kal = ang_vel_kal.reshape(-1,1)
    vel_s = np.hstack((velocity_kal, velocity_x_kal, velocity_y_kal, ang_vel_kal))
    print("Smoothed vel shape: (%2d, %2d)" %(int(vel_s.shape[0]), int(vel_s.shape[1])))
    return vel_s

def save_vel_smooth(subject:str, fold_path):
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
    vel_r = vel_resample(subject, fold_path)
    vel_s = vel_smooth(vel_r)
    df_vel_smooth = pd.DataFrame(data=vel_s)
    vel_smooth_path = fold_path + '/DataPreparation/Preprocessed_data/' + cricket_number + '/' + video_number +'_Velocity_Smooth.csv'
    df_vel_smooth.to_csv(path_or_buf=vel_smooth_path, header=False, index=False) # vel, x_vel, y_vel, ang_vel

''' 
DLC (direction and joint angle) data preprocessing: 
Direction data (120Hz) is converted from deg to vec, and all dlc data is cropped to synchronize with velocity data.
'''
# direction data is converted from deg to vec, for the consecutiveness of the time sequences
# oherwise, there will be unconsective when crickets are rotating around 0+ or 360-
def deg_to_vec (directon):
    direction_rad = directon * np.pi / 180
    direction_x = np.cos(direction_rad)
    direction_y = np.sin(direction_rad)
    return direction_x, direction_y

def dlc_crop(subject:str, fold_path):
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
        # "begin" means the number of data to be cropped at the beginning of the sequence
        # "end" means the number of data to be cropped at the end of the sequence
        [begin, end] = trail_details[f"T{subject}"]["video_synchronize_range"]
    # get absolute joint angle
    jointangle_absolute_path = fold_path +'/DataPreparation/Skeleton_data/' + cricket_number + '/PIC'+ video_number +'_JointAngle_Absolute.csv'
    jointangle_absolute = pd.read_csv(jointangle_absolute_path, header=0, usecols=[1,2,3,4,5,6,7,8,9,10,11,12])
    jointangle_absolute = np.array(jointangle_absolute)
    # get relative joint orientation
    jointangle_relative_path = fold_path + '/DataPreparation/Skeleton_data/' + cricket_number + '/PIC'+ video_number +'_JointAngle_Relative.csv'
    jointangle_relative = pd.read_csv(jointangle_relative_path, header=0, usecols=[1,2,3,4,5,6,7,8,9,10,11,12])
    jointangle_relative = np.array(jointangle_relative)
    # get direction
    ThetaCamBody_path = fold_path + '/DataPreparation/Skeleton_data/' + cricket_number + '/PIC'+ video_number +'_ThetaCamBody.csv'
    ThetaCamBody = pd.read_csv(ThetaCamBody_path, header=0, usecols=[0])
    ThetaCamBody = np.array(ThetaCamBody)
    # crop
    jointangle_absolute_crop = jointangle_absolute[begin:-end,:]
    jointangle_relative_crop = jointangle_relative[begin:-end,:]
    ThetaCamBody_crop = ThetaCamBody[begin:-end,:]
    # convert direction angle(deg) to vector
    Direction_x, Direction_y = deg_to_vec(ThetaCamBody_crop)
    ThetaCamBody_crop = ThetaCamBody_crop.reshape(-1,1)
    Direction_x = Direction_x.reshape(-1,1)
    Direction_y = Direction_y.reshape(-1,1)
    ThetaCamBody_crop = np.hstack((ThetaCamBody_crop, Direction_x, Direction_y))
    print("Video: %s, Original dlc shape: (%2d, %2d), Cropped dlc shape: (%2d, %2d)" %(video_number, int(jointangle_absolute.shape[0]), int(jointangle_absolute.shape[1]),
                                                                                                                                                                                    int(ThetaCamBody_crop.shape[0]), int(ThetaCamBody_crop.shape[1])))
    return jointangle_absolute_crop, jointangle_relative_crop, ThetaCamBody_crop

def save_dlc_crop(subject:str, fold_path):
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
    jointangle_absolute_crop, jointangle_relative_crop, ThetaCamBody_crop = dlc_crop(subject, fold_path)
    # absolute joint angle
    jointangle_absolute_crop = np.array(jointangle_absolute_crop)
    df_jointangle_absolute_crop = pd.DataFrame(data=jointangle_absolute_crop, columns=['ThC_LF','ThC_LM','ThC_LH','ThC_RF','ThC_RM','ThC_RH',
                                                                                                                                                                                             'FTi_LF','FTi_LM','FTi_LH','FTi_RF','FTi_RM','FTi_RH'])
    jointangle_absolute_crop_path = fold_path + '/DataPreparation/Preprocessed_data/' + cricket_number + '/' + video_number +'_JointAngle_Absolute_Crop.csv'
    df_jointangle_absolute_crop.to_csv(path_or_buf=jointangle_absolute_crop_path, header=True, index=True) 
    # relative joint angle
    jointangle_relative_crop = np.array(jointangle_relative_crop)
    df_jointangle_relative_crop = pd.DataFrame(data=jointangle_relative_crop, columns=['ThC_LF','ThC_LM','ThC_LH','ThC_RF','ThC_RM','ThC_RH',
                                                                                                                                                                                        'FTi_LF','FTi_LM','FTi_LH','FTi_RF','FTi_RM','FTi_RH'])
    jointangle_relative_crop_path = fold_path + '/DataPreparation/Preprocessed_data/' + cricket_number + '/' + video_number +'_JointAngle_Relative_Crop.csv'
    df_jointangle_relative_crop.to_csv(path_or_buf=jointangle_relative_crop_path, header=True, index=True) 
    # direction
    ThetaCamBody_crop = np.array(ThetaCamBody_crop)
    df_ThetaCamBody_crop = pd.DataFrame(data=ThetaCamBody_crop, columns=['ThetaCamBody','Direction_x','Direction_y'])
    ThetaCamBody_crop_path = fold_path + '/DataPreparation/Preprocessed_data/' + cricket_number + '/' + video_number +'_ThetaCamBody_Crop.csv'
    df_ThetaCamBody_crop.to_csv(path_or_buf=ThetaCamBody_crop_path, header=True, index=False) 

if __name__ == '__main__':
    fold_path = os.getcwd()
    # input the total number of subjects
    subjects = 33
    for i in range(subjects):
        i = i + 1
        if i < 10:
            subject_number = "0" + str(i)
        else:
            subject_number = str(i)
        data_preprocess(subject_number, fold_path)

