'''
This code will calculate the direction and leg joint angle of crickets from DeepLabCut results, and generate csv file(s) to record the data. 
It is supposed to be run in the path of /home/yuchen/Crickets_Walking_Motion_Prediction/expert_data_builder.
The DeepLabCut skeleton data is from /home/yuchen/Crickets_Walking_Motion_Prediction/DeepLabCut/videos/.
The generate skeleton and joint movement csv. data will be stored at /home/yuchen/Crickets_Walking_IRL/expert_data_builder/
original_skeleton_data/, and /home/yuchen/Crickets_Walking_IRL/expert_data_builder/movement/, respectively.
'''
import os
import json
import numpy as np
import pandas as pd

def get_skeleton(subject:str, fold_path):
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_name =  trail_details[f"T{subject}"]["cricket_name"]
        folder_date = trail_details[f"T{subject}"]["folder_date"]
        video_number = trail_details[f"T{subject}"]["video_number"]
        date = trail_details[f"T{subject}"]["date"]
        training_iter = trail_details[f"T{subject}"]["training_iter"]
    dlc_fold_path = fold_path + '/DeepLabCut/' + cricket_name + '-Yuchen-' + folder_date + '/videos/'
    skeleton_path = dlc_fold_path + 'PIC_' + video_number + 'DLC_resnet50_Cricket' + date + 'shuffle1_' + training_iter + '_skeleton.csv'
    skeleton = pd.read_csv(skeleton_path, header=[0,1], index_col=[0]) 
    df_skeleton = pd.DataFrame(data=skeleton) 
    return skeleton, df_skeleton

'''
The following function is to set the original skeleton data to the standard format.
'''

def delete_irrelevant_column(skeleton, df_skeleton):
    # delete length and likelihood of the first header columns
    df_skeleton.drop(columns=['length','likelihood'], axis=1, level=1, inplace=True)
    # delete assigned columns
    df_skeleton.drop(columns=['Head_Pro','Meso_Meta',
                            'LF0_LM0','LF1_LM0','LM0_LH0','LM1_LH0','LH1_LM0',
                            'RF0_RM0','RF1_RM0','RM0_RH0','RM1_RH0','RH1_RM0',
                            'LF0_LF2','LM0_LM2','LH0_LH2','RF0_RF2','RM0_RM2','RH0_RH2',
                            'Axis_Bar','Axis_Fix'], axis=1, level=0, inplace=True)
    df_skeleton = pd.DataFrame(data=df_skeleton.values, columns=['Pro_Meso',
                                    'LF1_LF0','LM1_LM0','LH1_LH0','RF1_RF0','RM1_RM0','RH1_RH0',                                                                                              
                                    'LF1_LF2','LM1_LM2','LH1_LH2','RF1_RF2','RM1_RM2','RH1_RH2'])
    return df_skeleton

def get_df_original_skeleton_data(subject:str, fold_path):
    skeleton, df_skeleton = get_skeleton(subject, fold_path)
    df_skeleton = delete_irrelevant_column(skeleton, df_skeleton)
    return df_skeleton

def save_original_skeleton_data(subject:str, fold_path):
    df_skeleton = get_df_original_skeleton_data(subject, fold_path)
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
    skeleton_path = fold_path + '/expert_data_builder/original_skeleton_data/' + cricket_number + '/PIC' + video_number + '_Skeleton.csv'
    df_skeleton.to_csv(path_or_buf = skeleton_path, header=True, index=True)

'''
The following function is to measure the joint movements.
'''
def get_inital_pos(df_skeleton):
    pro_meso = df_skeleton['Pro_Meso'].values
    inital_pos_left = []
    inital_pos_right = []
    for i in range(len(pro_meso)):
        if pro_meso[i] >= 90 and pro_meso[i] < 270:
            inital_pos_left_i = pro_meso[i] + 90
            inital_pos_right_i = pro_meso[i] - 90
        elif pro_meso[i] >= 270 and pro_meso[i] < 360:
            inital_pos_left_i = pro_meso[i] - 270
            inital_pos_right_i = pro_meso[i] - 90
        elif pro_meso[i] >= 0 and pro_meso[i] < 90:
            inital_pos_left_i = pro_meso[i] + 90
            inital_pos_right_i = pro_meso[i] + 270
        inital_pos_left.append(inital_pos_left_i)
        inital_pos_right.append(inital_pos_right_i)
    return np.array(inital_pos_left).reshape(-1,1), np.array(inital_pos_right).reshape(-1,1)

def get_reverse_orientation(data):
    reversed_data = []
    for i in range(len(data)):
        if data[i] >= 0 and data[i] < 180:
            reversed_data_i = data[i] + 180
        elif data[i] >= 180 and data[i] < 360:
            reversed_data_i = data[i] - 180
        reversed_data.append(reversed_data_i)
    return reversed_data

def fix_exceed_180(data):
    fixed_data = []
    for i in range(len(data)):
        if data[i] >= 180:
            fixed_data_i = - (360 - data[i])
            fixed_data.append(fixed_data_i)
        elif data[i] <= -180:
            fixed_data_i = 360 + data[i]
            fixed_data.append(fixed_data_i)
        else:
            fixed_data.append(data[i])
    return fixed_data

def get_joint_movement(subject:str, df_skeleton):
    skeleton = df_skeleton.to_numpy()
    inital_pos_left, inital_pos_right = get_inital_pos(df_skeleton)
    LF10, LM10, LH10, RF10, RM10, RH10 = skeleton[:,1].reshape(-1,1), skeleton[:,2].reshape(-1,1), skeleton[:,3].reshape(-1,1),\
                                                                                    skeleton[:,4].reshape(-1,1), skeleton[:,5].reshape(-1,1), skeleton[:,6].reshape(-1,1)
    LF12, LM12, LH12, RF12, RM12, RH12 = skeleton[:,7].reshape(-1,1), skeleton[:,8].reshape(-1,1), skeleton[:,9].reshape(-1,1),\
                                                                                    skeleton[:,10].reshape(-1,1), skeleton[:,11].reshape(-1,1), skeleton[:,12].reshape(-1,1)
    # reverse 180 deg for the femur orientation
    LF01 = get_reverse_orientation(LF10); LM01 = get_reverse_orientation(LM10); LH01 = get_reverse_orientation(LH10)
    RF01 = get_reverse_orientation(RF10); RM01 = get_reverse_orientation(RM10); RH01 = get_reverse_orientation(RH10)
    # calculate the ThC joint movement
    ThC_LF = LF01 - inital_pos_left; ThC_RF = RF01 - inital_pos_right
    ThC_LM = LM01 - inital_pos_left; ThC_RM = RM01 - inital_pos_right
    ThC_LH = LH01 - inital_pos_left; ThC_RH = RH01 - inital_pos_right
    # calculate the FTi joint movement
    FTi_LF = LF12 - LF01; FTi_RF = RF12 - RF01
    FTi_LM = LM12 - LM01; FTi_RM = RM12 - RM01
    FTi_LH = LH12 - LH01; FTi_RH = RH12 - RH01
    # fix the joint movement that exceed plus/minus 180 deg
    ThC_LF = fix_exceed_180(ThC_LF); ThC_LM = fix_exceed_180(ThC_LM); ThC_LH = fix_exceed_180(ThC_LH)
    ThC_RF = fix_exceed_180(ThC_RF); ThC_RM = fix_exceed_180(ThC_RM); ThC_RH = fix_exceed_180(ThC_RH)
    FTi_LF = fix_exceed_180(FTi_LF); FTi_LM = fix_exceed_180(FTi_LM); FTi_LH = fix_exceed_180(FTi_LH)
    FTi_RF = fix_exceed_180(FTi_RF); FTi_RM = fix_exceed_180(FTi_RM); FTi_RH = fix_exceed_180(FTi_RH)
    joint_movement = np.hstack((ThC_LF, ThC_LM, ThC_LH, ThC_RF, ThC_RM, ThC_RH,
                                FTi_LF, FTi_LM, FTi_LH, FTi_RF, FTi_RM, FTi_RH))
    # crop the joint movement data to synchronize with the velocity data
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        [begin, end] = trail_details[f"T{subject}"]["video_synchronize_range"]
        joint_movement = joint_movement[begin:-end,:]
    df_joint_movement = pd.DataFrame(data=joint_movement, columns=['ThC_LF', 'ThC_LM', 'ThC_LH', 'ThC_RF', 'ThC_RM', 
                                                                   'ThC_RH', 'FTi_LF', 'FTi_LM', 'FTi_LH', 'FTi_RF', 'FTi_RM', 'FTi_RH'])
    return df_joint_movement

def save_joint_movement(subject:str, fold_path):
    df_skeleton = get_df_original_skeleton_data(subject, fold_path)
    df_joint_movement = get_joint_movement(subject, df_skeleton)
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
    joint_movement_path = fold_path + '/expert_data_builder/movement/' + cricket_number + '/PIC' + video_number + '_Joint_movement.csv'
    df_joint_movement.to_csv(path_or_buf = joint_movement_path, header=True, index=True)

'''
The following function is to set the heading direction 
(from counterclockwise to clockwise).
'''
def get_heading_direction(subject:str, fold_path):
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
    direction_path = fold_path + '/DataPreparation/Preprocessed_data/' + cricket_number + '/' + video_number + '_ThetaCamBody_Crop.csv'
    direction = pd.read_csv(direction_path, header=0, usecols=[0]).to_numpy()
    # revise the data from counterclockwise to clockwise
    direction = 360 - direction
    # save the heading direction
    save_direction_path = fold_path + '/expert_data_builder/movement/' + cricket_number + '/PIC' + video_number + '_Heading_direction.csv'
    pd.DataFrame(data=direction, columns=['Heading_direction']).to_csv(path_or_buf = save_direction_path, header=True, index=True)

'''
The following function is to calculate the trajectory of the cricket.
(from counterclockwise to clockwise).
'''
def get_trajectory(subject:str, fold_path):
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
    vel_path = fold_path + '/DataPreparation/Preprocessed_data/' + cricket_number + '/' + video_number + '_Velocity_Smooth.csv'
    vel = pd.read_csv(vel_path, header=None, usecols=[1,2]).to_numpy() # vel.x and vel.y
    # scale velocity: mm/s
    vel = vel * 0.224077
    # frequency: 119.88(120)Hz
    # trajectory calculation
    traj_x = [0]; x=0
    traj_y = [0]; y=0
    for i in range(1, len(vel)):
        x = x + vel[i][0]*1/119.88/1000
        y = y + vel[i][1]*1/119.88/1000
        traj_x.append(x)
        traj_y.append(y)
    traj = np.array([traj_x, traj_y]).reshape(-1, 2)
    # save the trajectory
    save_traj_path = fold_path + '/expert_data_builder/movement/' + cricket_number + '/PIC' + video_number + '_Trajectory.csv'
    pd.DataFrame(data=traj, columns=['x', 'y']).to_csv(path_or_buf = save_traj_path, header=True, index=True)

if __name__ == '__main__':
    # return to the root fold path
    fold_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # input the total number of subjects
    subjects = 33
    for i in range(subjects):
        i = i + 1
        if i < 10:
            subject_number = "0" + str(i)
        else:
            subject_number = str(i)
        # skip c16,c18,c20 
        with open("trail_details.json", "r") as f:
            config = json.load(f)
            c_number =  config[f"T{subject_number}"]["cricket_number"]
        if c_number in ['c16', 'c18', 'c20']:
            continue
        else:
            save_original_skeleton_data(subject_number, fold_path)
            save_joint_movement(subject_number, fold_path)
            get_heading_direction(subject_number, fold_path)
            get_trajectory(subject_number, fold_path)
