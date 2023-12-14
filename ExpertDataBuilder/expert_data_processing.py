'''
This code will calculate the direction and leg joint angle of crickets from DeepLabCut results, and generate csv file(s) to record the data. 
The DeepLabCut skeleton data is from /DeepLabCut/videos/.
The generate skeleton analysis csv. data will be stored at /DataPreparation/Skeleton_data/.
'''
import os
import json
import math
import numpy as np
import pandas as pd

def skeleton_analysis(subject:str, fold_path):
    save_original_skeleton_data(subject, fold_path)
    save_absolute_JointAngle(subject, fold_path)
    save_relative_JointAngle(subject, fold_path)

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
    skeleton = pd.read_csv(skeleton_path, header=[1], index_col=[0]) 
    df_skeleton = pd.DataFrame(data=skeleton) 
    return skeleton, df_skeleton

''' 
Extract information from the skeleton of DeepLabCut data 
and delete irrelavent data for further use.
'''
def delete_irrelevant_column(skeleton, df_skeleton):
    n_col = skeleton.shape[1] 
    for i in range(n_col):  
        # delete the "length" columns
        if (i % 3 == 0):  
            j = i/3
            if (j == 0):
                df_skeleton.drop(columns=['length'], axis=1, inplace=True)
            else:
                df_skeleton.drop(columns=['length.'+ str(int(j))], axis=1, inplace=True)
        # delete the "likelihood" columns
        elif (i % 3 == 2):  
            j = (i-2)/3
            if (j == 0):
                df_skeleton.drop(columns=['likelihood'], axis=1, inplace=True)
            else:
                df_skeleton.drop(columns=['likelihood.'+ str(int(j))], axis=1, inplace=True)
    # delete assigned columns
    df_skeleton.drop(columns=['Head_Pro','Meso_Meta',
                            'LF0_LM0','LF1_LM0','LM0_LH0','LM1_LH0','LH1_LM0',
                            'RF0_RM0','RF1_RM0','RM0_RH0','RM1_RH0','RH1_RM0',
                            'LF0_LF2','LM0_LM2','LH0_LH2','RF0_RF2','RM0_RM2','RH0_RH2',
                            'Axis_Bar','Axis_Fix'], axis=1, inplace=True)
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
    skeleton_path = fold_path + '/ExpertDataBuilder/Original_skeleton_data/' + cricket_number + '/PIC' + video_number + '_Skeleton.csv'
    df_skeleton.to_csv(path_or_buf = skeleton_path, header=True, index=True)

'''
Get leg joint angles(absolute value) of the cricket.
'''
def calculate_joint(a,b,c):
    temp = (np.square(b)+np.square(c)-np.square(a))/(2*b*c)
    joint = np.arccos(temp)
    joint = joint*180/math.pi
    return joint

def get_absolute_JointAngle(subject:str, col_oLF10=3):
    df_skeleton = get_df_skeleton_absolute_angles(subject, fold_path)
    skeleton = df_skeleton.values
    # lengths for left ThC 
    l_LF1_LF0 = skeleton[:,col_oLF10]
    l_LF0_LM0 = skeleton[:,col_oLF10+1]
    l_LF1_LM0 = skeleton[:,col_oLF10+2]
    l_LM1_LM0 = skeleton[:,col_oLF10+3]
    l_LM0_LH0 = skeleton[:,col_oLF10+4]
    l_LM1_LH0 = skeleton[:,col_oLF10+5]
    l_LH1_LH0 = skeleton[:,col_oLF10+6]
    l_LH1_LM0 = skeleton[:,col_oLF10+7]
    # lengths for right ThC 
    l_RF1_RF0 = skeleton[:,col_oLF10+8]
    l_RF0_RM0 = skeleton[:,col_oLF10+9]
    l_RF1_RM0 = skeleton[:,col_oLF10+10]
    l_RM1_RM0 = skeleton[:,col_oLF10+11]
    l_RM0_RH0 = skeleton[:,col_oLF10+12]
    l_RM1_RH0 = skeleton[:,col_oLF10+13]
    l_RH1_RH0 = skeleton[:,col_oLF10+14]
    l_RH1_RM0 = skeleton[:,col_oLF10+15]
    # lengths for left FTi
    l_LF1_LF2 = skeleton[:,col_oLF10+16]
    l_LF0_LF2 = skeleton[:,col_oLF10+17]
    l_LM1_LM2 = skeleton[:,col_oLF10+18]
    l_LM0_LM2 = skeleton[:,col_oLF10+19]
    l_LH1_LH2 = skeleton[:,col_oLF10+20]
    l_LH0_LH2 = skeleton[:,col_oLF10+21]
    # right legs length
    l_RF1_RF2 = skeleton[:,col_oLF10+22]
    l_RF0_RF2 = skeleton[:,col_oLF10+23]
    l_RM1_RM2 = skeleton[:,col_oLF10+24]
    l_RM0_RM2 = skeleton[:,col_oLF10+25]
    l_RH1_RH2 = skeleton[:,col_oLF10+26]
    l_RH0_RH2 = skeleton[:,col_oLF10+27]
    # calculate ThC
    ThC_LF = calculate_joint(l_LF1_LM0,l_LF1_LF0,l_LF0_LM0).reshape(-1,1)
    ThC_LM = calculate_joint(l_LM1_LH0,l_LM1_LM0,l_LM0_LH0).reshape(-1,1)
    ThC_LH = calculate_joint(l_LH1_LM0,l_LH1_LH0,l_LM0_LH0).reshape(-1,1)
    ThC_LH = 180 - ThC_LH 
    ThC_RF = calculate_joint(l_RF1_RM0,l_RF1_RF0,l_RF0_RM0).reshape(-1,1)
    ThC_RM = calculate_joint(l_RM1_RH0,l_RM1_RM0,l_RM0_RH0).reshape(-1,1)
    ThC_RH = calculate_joint(l_RH1_RM0,l_RH1_RH0,l_RM0_RH0).reshape(-1,1)
    ThC_RH = 180 - ThC_RH
    # calculate FTi
    FTi_LF = calculate_joint(l_LF0_LF2,l_LF1_LF0,l_LF1_LF2).reshape(-1,1)
    FTi_LM = calculate_joint(l_LM0_LM2,l_LM1_LM0,l_LM1_LM2).reshape(-1,1)
    FTi_LH = calculate_joint(l_LH0_LH2,l_LH1_LH0,l_LH1_LH2).reshape(-1,1)
    FTi_RF = calculate_joint(l_RF0_RF2,l_RF1_RF0,l_RF1_RF2).reshape(-1,1)
    FTi_RM = calculate_joint(l_RM0_RM2,l_RM1_RM0,l_RM1_RM2).reshape(-1,1)
    FTi_RH = calculate_joint(l_RH0_RH2,l_RH1_RH0,l_RH1_RH2).reshape(-1,1)
    # organize the data
    JointAngle = np.hstack((ThC_LF,ThC_LM,ThC_LH,ThC_RF,ThC_RM,ThC_RH,FTi_LF,FTi_LM,FTi_LH,FTi_RF,FTi_RM,FTi_RH))  
    df_JointAngle = pd.DataFrame(data=JointAngle, columns=['ThC_LF','ThC_LM','ThC_LH','ThC_RF','ThC_RM','ThC_RH','FTi_LF','FTi_LM','FTi_LH','FTi_RF','FTi_RM','FTi_RH'])
    return df_JointAngle

def save_absolute_JointAngle(subject:str, fold_path):
    df_JointAngle = get_absolute_JointAngle(subject, col_oLF10=3)
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
    JointAngle_path = fold_path + '/DataPreparation/Skeleton_data/' + cricket_number + '/PIC' + video_number + '_JointAngle_Absolute.csv'
    df_JointAngle.to_csv(path_or_buf=JointAngle_path, header=True, index=True)

'''
Get leg joint angles(relative value) of the cricket.
'''
def get_relative_JointAngle(subject:str, col_oLF10=3):
    df_skeleton = get_df_skeleton_relative_angles(subject, fold_path)
    skeleton = df_skeleton.values
    # body-part and hip-leg orientation
    o_HeadPro = skeleton[:,0]
    o_ProMeso = skeleton[:,1]
    o_MesoMeta = skeleton[:,2]
    # left legs orientation
    o_LF10 = skeleton[:,col_oLF10]
    o_LF12 = skeleton[:,col_oLF10+16]
    o_LM10 = skeleton[:,col_oLF10+3]
    o_LM12 = skeleton[:,col_oLF10+18]
    o_LH10 = skeleton[:,col_oLF10+6]
    o_LH12 = skeleton[:,col_oLF10+20]
    # right legs orientation
    o_RF10 = skeleton[:,col_oLF10+8]
    o_RF12 = skeleton[:,col_oLF10+22]
    o_RM10 = skeleton[:,col_oLF10+11]
    o_RM12 = skeleton[:,col_oLF10+24]
    o_RH10 = skeleton[:,col_oLF10+14]
    o_RH12 = skeleton[:,col_oLF10+26]
    # calculate_ThC
    ThC_LF = o_HeadPro - o_LF10
    ThC_LM = o_ProMeso - o_LM10
    ThC_LH = o_MesoMeta - o_LH10
    ThC_RF = o_HeadPro - o_RF10
    ThC_RM = o_ProMeso - o_RM10
    ThC_RH = o_MesoMeta - o_RH10
    # calculate_FTi
    FTi_LF = o_LF10 - o_LF12
    FTi_LM = o_LM10 - o_LM12
    FTi_LH = o_LH10 - o_LH12
    FTi_RF = o_RF10 - o_RF12
    FTi_RM = o_RM10 - o_RM12
    FTi_RH = o_RH10 - o_RH12
    # reshape
    ThC_LF = ThC_LF.reshape(-1,1)
    ThC_LM = ThC_LM.reshape(-1,1)
    ThC_LH = ThC_LH.reshape(-1,1)
    ThC_RF = ThC_RF.reshape(-1,1)
    ThC_RM = ThC_RM.reshape(-1,1)
    ThC_RH = ThC_RH.reshape(-1,1)
    FTi_LF = FTi_LF.reshape(-1,1)
    FTi_LM = FTi_LM.reshape(-1,1)
    FTi_LH = FTi_LH.reshape(-1,1)
    FTi_RF = FTi_RF.reshape(-1,1)
    FTi_RM = FTi_RM.reshape(-1,1)
    FTi_RH = FTi_RH.reshape(-1,1)
    # organize the data
    JointAngle = np.hstack((ThC_LF,ThC_LM,ThC_LH,ThC_RF,ThC_RM,ThC_RH,FTi_LF,FTi_LM,FTi_LH,FTi_RF,FTi_RM,FTi_RH))  
    df_JointAngle = pd.DataFrame(data=JointAngle, columns=['ThC_LF','ThC_LM','ThC_LH','ThC_RF','ThC_RM','ThC_RH','FTi_LF','FTi_LM','FTi_LH','FTi_RF','FTi_RM','FTi_RH'])
    return df_JointAngle

def save_relative_JointAngle(subject:str, fold_path):
    df_JointAngle = get_relative_JointAngle(subject, col_oLF10=3)
    with open("trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
    JointAngle_path = fold_path + '/DataPreparation/Skeleton_data/' + cricket_number + '/PIC' + video_number + '_JointAngle_Relative.csv'
    df_JointAngle.to_csv(path_or_buf=JointAngle_path, header=True, index=True)

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
        skeleton_analysis(subject_number, fold_path)
