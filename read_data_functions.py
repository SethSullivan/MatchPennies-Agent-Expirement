import numpy as np
import dill
import os
import importlib
import pandas as pd
import sys
import itertools
sys.path.insert(0,r'D:\OneDrive - University of Delaware - o365\Desktop\MatchPennies-Agent-Expirement')
import Subject_Object_v2
importlib.reload(Subject_Object_v2)

def pickle_subject_objects(subject_list,load_path):
    subject_objects = []
    for i,subname in enumerate(subject_list):
        print(subname)
        subject_object = dill.load(open(load_path + f'\\{subname}\\{subname}_object.pkl', 'rb'))
        subject_objects.append(subject_object)
    return subject_objects

def generate_subject_objects(experiment):
    if True:
        experiment = experiment

        if experiment == 'Exp1':
            coincidence_trials = 50
            interval_trials = 50
            reaction_trials = 50
            reaction_blocks = 3
            trial_time = 5000
            task_blocks = 6
            task_trials = 80
        elif experiment == 'Exp2':
            coincidence_trials = 50
            interval_trials = 50
            reaction_trials = 100
            reaction_blocks = 3
            trial_time = 8000
            task_blocks = 4
            task_trials = 80
    if True:
        # Fields pull and pull list
        PATH = f'D:\\OneDrive - University of Delaware - o365\\Subject_Data\\MatchPennies_Agent_{experiment}'
        os.chdir(PATH)
        figures_pull_list = []
        fields_pull = []
        with open(PATH+"\\Figures_Pull_List.txt", "r") as pull_file:
            figures_pull_list = pull_file.read().splitlines()
        with open(PATH+"\\Fields_Pull.txt", "r") as fields_pull:
            fields_pull = fields_pull.read().splitlines()
        num_subjects = len(figures_pull_list)
    #* Load control data
    if True:
        # ---------------Controls-------------------------
        reaction_trial_start          = np.zeros((num_subjects, reaction_blocks, reaction_trials))*np.nan
        reaction_filenames            = np.empty((num_subjects, reaction_blocks,reaction_trials),dtype = object)
        agent_reaction_leave_time     = np.zeros((num_subjects, reaction_blocks,reaction_trials))*np.nan
        agent_reaction_decision_array = np.empty((num_subjects, reaction_blocks,reaction_trials))*np.nan
        reaction_trial_type_array     = np.zeros((num_subjects, reaction_blocks,reaction_trials))*np.nan
        reaction_xypos_data           = np.zeros((num_subjects, reaction_blocks,reaction_trials, trial_time,2))*np.nan
        reaction_dist_data            = np.zeros((num_subjects, reaction_blocks,reaction_trials, trial_time))*np.nan
        reaction_xyvelocity_data      = np.zeros((num_subjects, reaction_blocks,reaction_trials, trial_time,2))*np.nan
        reaction_speed_data           = np.zeros((num_subjects, reaction_blocks,reaction_trials, trial_time))*np.nan
        # reaction_xyforce_data         = np.zeros((num_subjects, reaction_blocks,reaction_trials, trial_time,2))*np.nan
        # reaction_force_data           = np.zeros((num_subjects, reaction_blocks,reaction_trials, trial_time))*np.nan 

        coincidence_trial_start                       = np.zeros((num_subjects, coincidence_trials))*np.nan
        coincidence_reach_time                        = np.zeros((num_subjects, coincidence_trials))*np.nan
        interval_trial_start                          = np.zeros((num_subjects, interval_trials))*np.nan
        interval_reach_time                           = np.zeros((num_subjects, interval_trials))*np.nan

        for i in range(num_subjects):
            subname = figures_pull_list[i]
            data_path = PATH+f'\\Subjects_Analyzed\\{subname}\\'
            reaction_trial_start[i,...]          = dill.load(open(data_path + f'{subname}_reaction_trial_start.pkl','rb'))          
            reaction_trial_type_array[i,...]          = dill.load(open(data_path + f'{subname}_reaction_trial_type_array.pkl','rb'))          
            # reaction_filenames[i,...]            = dill.load(open(data_path + f'{subname}_reaction_filenames.pkl','rb'))            
            agent_reaction_leave_time[i,...]     = dill.load(open(data_path + f'{subname}_agent_reaction_leave_time.pkl','rb'))  
            agent_reaction_decision_array[i,...] = dill.load(open(data_path + f'{subname}_agent_reaction_decision_array.pkl','rb')) 
            reaction_xypos_data[i,...]           = dill.load(open(data_path + f'{subname}_reaction_xypos_data.pkl','rb'))           
            reaction_dist_data[i,...]            = dill.load(open(data_path + f'{subname}_reaction_dist_data.pkl','rb'))            
            reaction_xyvelocity_data[i,...]      = dill.load(open(data_path + f'{subname}_reaction_xyvelocity_data.pkl','rb'))      
            reaction_speed_data[i,...]           = dill.load(open(data_path + f'{subname}_reaction_speed_data.pkl','rb'))            
            interval_trial_start[i,:]            = dill.load(open(data_path + f'{subname}_interval_trial_start.pkl', 'rb'))
            interval_reach_time[i,:]             = dill.load(open(data_path + f'{subname}_interval_reach_time.pkl', 'rb'))
            coincidence_trial_start[i,:]         = dill.load(open(data_path + f'{subname}_coincidence_trial_start.pkl', 'rb'))
            coincidence_reach_time[i,:]          = dill.load(open(data_path + f'{subname}_coincidence_reach_time.pkl', 'rb'))
    #* Load Task Data
    if True:
        path1 = PATH+'\\'+'Sub1_Task'
        task_df = pd.read_csv(path1+f'\\Sub1_TaskTrial_Table.csv')
        task_df = task_df.loc[task_df['Condition type']==3] # Only get the task condition 
        num_trials = int(task_df.iloc[-1]['Block_Step']) # number of trials in each block
        num_blocks = int(task_df.iloc[-1]['Block_Row'])
        tot_trials = int(num_trials*num_blocks)
        trial_time = int(task_df.iloc[0]['Condition time'])
        trial_time = 2000
        # ---------------Controls-------------------------
        task_trial_start          = np.zeros((num_subjects, task_blocks, task_trials))*np.nan
        task_filenames            = np.empty((num_subjects, task_blocks,task_trials),dtype = object)
        agent_task_leave_time     = np.zeros((num_subjects, task_blocks,task_trials))*np.nan
        agent_task_decision_array = np.empty((num_subjects, task_blocks,task_trials))*np.nan
        task_xypos_data           = np.zeros((num_subjects, task_blocks,task_trials, trial_time,2))*np.nan
        task_dist_data            = np.zeros((num_subjects, task_blocks,task_trials, trial_time))*np.nan
        task_xyvelocity_data      = np.zeros((num_subjects, task_blocks,task_trials, trial_time,2))*np.nan
        task_speed_data           = np.zeros((num_subjects, task_blocks,task_trials, trial_time))*np.nan 
        
        for i in range(num_subjects):
            subname = figures_pull_list[i]
            data_path = PATH+f'\\Subjects_Analyzed\\{subname}\\'
            task_trial_start[i,...]          = dill.load(open(data_path + f'{subname}_task_trial_start.pkl','rb'))                      
            agent_task_leave_time[i,...]     = dill.load(open(data_path + f'{subname}_agent_task_leave_time.pkl','rb'))  
            agent_task_decision_array[i,...] = dill.load(open(data_path + f'{subname}_agent_task_decision_array.pkl','rb')) 
            task_xypos_data[i,...]           = dill.load(open(data_path + f'{subname}_task_xypos_data.pkl','rb'))           
            task_dist_data[i,...]            = dill.load(open(data_path + f'{subname}_task_dist_data.pkl','rb'))            
            task_xyvelocity_data[i,...]      = dill.load(open(data_path + f'{subname}_task_xyvelocity_data.pkl','rb'))      
            task_speed_data[i,...]           = dill.load(open(data_path + f'{subname}_task_speed_data.pkl','rb'))           
            
    #* Generate Subject Objects
    if True:
        path1 = PATH+'\\'+'Sub1_Task'
        task_df = pd.read_csv(path1+f'\\Sub1_TaskTrial_Table.csv')
        task_df = task_df.loc[task_df['Condition type']==3] # Only get the task condition 
        num_trials = int(task_df.iloc[-1]['Block_Step']) # number of trials in each block
        if experiment == 'Exp2':
            num_blocks = int(task_df.iloc[-1]['Block_Row'])
            num_reaction_trials = 100
        else:
            num_blocks = int(task_df.iloc[-1]['Block_Row']/2)
            num_reaction_trials = 50
        tot_trials = int(num_trials*num_blocks)
        trial_time = int(task_df.iloc[0]['Condition time']) + 500
        task_df_columns = len(fields_pull)
        trial_table = np.empty((num_subjects, tot_trials, 4), int)

        data_path = 'Subjects_Analyzed\\'
        subject_objects = []
        for i in range(num_subjects):
            subname = figures_pull_list[i]
            print(subname)
            subject_object = Subject_Object_v2.Subject(
                subject = subname,experiment = experiment, num_task_trials_initial = num_trials, num_task_blocks = num_blocks, num_reaction_blocks = 3, num_reaction_trials = num_reaction_trials,num_timing_trials = 50,
                reaction_trial_start = reaction_trial_start[i],
                reaction_xypos_data = reaction_xypos_data[i],reaction_dist_data = reaction_dist_data[i],
                reaction_xyvelocity_data = reaction_xyvelocity_data[i],reaction_speed_data = reaction_speed_data[i],reaction_trial_type_array = reaction_trial_type_array[i], agent_reaction_decision_array = agent_reaction_decision_array[i],
                agent_reaction_leave_time = agent_reaction_leave_time[i], 
                
                task_xypos_data = task_xypos_data[i], task_dist_data = task_dist_data[i], task_xyvelocity_data = task_xyvelocity_data[i], task_speed_data = task_speed_data[i],
                interval_trial_start = interval_trial_start[i], interval_reach_time = interval_reach_time[i], coincidence_trial_start = coincidence_trial_start[i], coincidence_reach_time = coincidence_reach_time[i],
                agent_task_leave_time = agent_task_leave_time[i], agent_task_decision_array = agent_task_decision_array[i],
                                        )
            subject_objects.append(subject_object)
            
    return subject_objects