'''
This code will generate the dataset of cricket behavior experiments from preprocessed data.
The generate preprocessed csv. data will be stored at /Dataset/.
'''
import os
import json
import numpy as np
import pandas as pd

def generate_dataset(subject:str, fold_path):
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
    JointAngle_Absolute_path = fold_path + '/DataPreparation/Preprocessed_data/' + cricket_number + '/' + video_number +'_JointAngle_Absolute_Crop.csv'
    JointAngle_Relative_path = fold_path + '/DataPreparation/Preprocessed_data/' + cricket_number + '/' + video_number +'_JointAngle_Relative_Crop.csv'
    direction_path = fold_path + '/DataPreparation/Preprocessed_data/' + cricket_number + '/' + video_number +'_ThetaCamBody_Crop.csv'
    vel_path = fold_path + '/DataPreparation/Preprocessed_data/' + cricket_number + '/' + video_number +'_Velocity_Smooth.csv'
    
    JointAngle_Absolute = pd.read_csv(JointAngle_Absolute_path, header=0, usecols=[1,2,3,4,5,6,7,8,9,10,11,12])
    JointAngle_Relative = pd.read_csv(JointAngle_Relative_path, header=0, usecols=[1,2,3,4,5,6,7,8,9,10,11,12])
    direction = pd.read_csv(direction_path, header=0, usecols=[0,1,2]) 
    vel = pd.read_csv(vel_path, header=None, usecols=[0,1,2,3])

    JointAngle_Absolute = np.array(JointAngle_Absolute)
    JointAngle_Relative = np.array(JointAngle_Relative)
    direction = np.array(direction)
    vel = np.array(vel)
    
    #scale parameter: mm/s
    vel = vel * 0.224077 

    dataset_absolute = np.hstack((JointAngle_Absolute, direction, vel))
    df_dataset_absolute = pd.DataFrame(data=dataset_absolute,columns=['ThC_LF','ThC_LM','ThC_LH','ThC_RF','ThC_RM','ThC_RH',
                                                    'FTi_LF','FTi_LM','FTi_LH','FTi_RF','FTi_RM','FTi_RH',
                                                    'Direction','Direction_x','Direction_y',
                                                    'Vel','Vel.x','Vel.y','Ang.Vel'])
    dataset_relative = np.hstack((JointAngle_Relative, direction, vel))
    df_dataset_relative = pd.DataFrame(data=dataset_relative,columns=['ThC_LF','ThC_LM','ThC_LH','ThC_RF','ThC_RM','ThC_RH',
                                                    'FTi_LF','FTi_LM','FTi_LH','FTi_RF','FTi_RM','FTi_RH',
                                                    'Direction','Direction_x','Direction_y',
                                                    'Vel','Vel.x','Vel.y','Ang.Vel'])
    return df_dataset_absolute, df_dataset_relative

def save_dataset(subject:str, fold_path):
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
    df_dataset_absolute, df_dataset_relative = generate_dataset(subject, fold_path)
    dataset_absolute_path = fold_path + '/Dataset/' + cricket_number + '/' + video_number +'_Dataset_Absolute.csv'
    df_dataset_absolute.to_csv(path_or_buf=dataset_absolute_path, header=True, index=True)
    dataset_relative_path = fold_path + '/Dataset/' + cricket_number + '/' + video_number +'_Dataset_Relative.csv'
    df_dataset_relative.to_csv(path_or_buf=dataset_relative_path, header=True, index=True)


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
        save_dataset(subject_number, fold_path)