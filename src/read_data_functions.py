import numpy as np
import dill
import os
import importlib
import pandas as pd
import sys
import itertools
sys.path.insert(0,r'D:\OneDrive - University of Delaware - o365\Desktop\MatchPennies-Agent-Expirement')
from Subject_Object_v3 import SubjectBuilder
import Subject_Object_v2

def pickle_subject_objects(subject_list,load_path):
    '''
    Deprecated, generating subject objects is faster than storing the processed objects
    '''
    subject_objects = []
    for i,subname in enumerate(subject_list):
        print(subname)
        subject_object = dill.load(open(load_path + f'\\{subname}\\{subname}_object.pkl', 'rb'))
        subject_objects.append(subject_object)
    return subject_objects

def generate_subject_objects(experiment):
    #* Experiment specific Constants
    if True:
        experiment = experiment

        if experiment == 'Exp1':
            COINCIDENCE_TRIALS = 50
            INTERVAL_TRIALS = 50
            REACTION_TRIALS = 50
            REACTION_BLOCKS = 3
            REACTION_TRIAL_TIME = 5000
            TASK_BLOCKS = 6
            TASK_TRIALS = 80
        elif experiment == 'Exp2':
            COINCIDENCE_TRIALS = 50
            INTERVAL_TRIALS = 50
            REACTION_TRIALS = 100
            REACTION_BLOCKS = 3
            REACTION_TRIAL_TIME = 8000
            TASK_BLOCKS = 4
            TASK_TRIALS = 80
        TIMING_TRIALS = 50
        TRIAL_TIME = 2000
        
    #* Get pull list and num subjects
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
        NUM_SUBJECTS = len(figures_pull_list)
    
    #* Load control data
    if True:
        # ---------------Controls-------------------------
        reaction_trial_start          = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS, REACTION_TRIALS))*np.nan
        reaction_filenames            = np.empty((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS),dtype = object)
        agent_reaction_leave_time     = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS))*np.nan
        agent_reaction_decision_array = np.empty((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS))*np.nan
        reaction_trial_type_array     = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS))*np.nan
        reaction_xypos_data           = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS, REACTION_TRIAL_TIME,2))*np.nan
        reaction_dist_data            = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS, REACTION_TRIAL_TIME))*np.nan
        reaction_xyvelocity_data      = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS, REACTION_TRIAL_TIME,2))*np.nan
        reaction_speed_data           = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS, REACTION_TRIAL_TIME))*np.nan
        # reaction_xyforce_data         = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS, REACTION_TRIAL_TIME,2))*np.nan
        # reaction_force_data           = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS, REACTION_TRIAL_TIME))*np.nan 

        coincidence_trial_start                       = np.zeros((NUM_SUBJECTS, COINCIDENCE_TRIALS))*np.nan
        coincidence_reach_time                        = np.zeros((NUM_SUBJECTS, COINCIDENCE_TRIALS))*np.nan
        interval_trial_start                          = np.zeros((NUM_SUBJECTS, INTERVAL_TRIALS))*np.nan
        interval_reach_time                           = np.zeros((NUM_SUBJECTS, INTERVAL_TRIALS))*np.nan

        for i in range(NUM_SUBJECTS):
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
        # ---------------Controls-------------------------
        task_trial_start          = np.zeros((NUM_SUBJECTS, TASK_BLOCKS, TASK_TRIALS))*np.nan
        task_filenames            = np.empty((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS),dtype = object)
        agent_task_leave_time     = np.zeros((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS))*np.nan
        agent_task_decision_array = np.empty((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS))*np.nan
        task_xypos_data           = np.zeros((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS, TRIAL_TIME,2))*np.nan
        task_dist_data            = np.zeros((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS, TRIAL_TIME))*np.nan
        task_xyvelocity_data      = np.zeros((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS, TRIAL_TIME,2))*np.nan
        task_speed_data           = np.zeros((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS, TRIAL_TIME))*np.nan 
        
        for i in range(NUM_SUBJECTS):
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
        data_path = 'Subjects_Analyzed\\'
        subject_objects = []
        for i in range(NUM_SUBJECTS):
            subname = figures_pull_list[i]
            print(subname)
            subject_object = Subject_Object_v2.Subject(
                subject = subname,experiment = experiment, num_task_trials_initial = TASK_TRIALS, num_task_blocks = TASK_BLOCKS, num_reaction_blocks = REACTION_BLOCKS, num_reaction_trials = REACTION_TRIALS,num_timing_trials = TIMING_TRIALS,
                reaction_trial_start = reaction_trial_start[i], reaction_xypos_data = reaction_xypos_data[i],reaction_dist_data = reaction_dist_data[i],
                reaction_xyvelocity_data = reaction_xyvelocity_data[i],reaction_speed_data = reaction_speed_data[i],reaction_trial_type_array = reaction_trial_type_array[i], agent_reaction_decision_array = agent_reaction_decision_array[i],
                agent_reaction_leave_time = agent_reaction_leave_time[i], 
                
                task_xypos_data = task_xypos_data[i], task_dist_data = task_dist_data[i], task_xyvelocity_data = task_xyvelocity_data[i], task_speed_data = task_speed_data[i],
                interval_trial_start = interval_trial_start[i], interval_reach_time = interval_reach_time[i], coincidence_trial_start = coincidence_trial_start[i], coincidence_reach_time = coincidence_reach_time[i],
                agent_task_leave_time = agent_task_leave_time[i], agent_task_decision_array = agent_task_decision_array[i],
                                        )
            subject_objects.append(subject_object)
    assert TRIAL_TIME == 2000    
    return subject_objects


def generate_subject_object_v3(experiment, select_trials='All Trials'):
    #* Experiment specific Constants
    if True:
        experiment = experiment

        if experiment == 'Exp1':
            COINCIDENCE_TRIALS = 50
            INTERVAL_TRIALS = 50
            REACTION_TRIALS = 50
            REACTION_BLOCKS = 3
            REACTION_TRIAL_TIME = 5000
            TASK_BLOCKS = 6
            TASK_TRIALS = 80
        elif experiment == 'Exp2':
            COINCIDENCE_TRIALS = 50
            INTERVAL_TRIALS = 50
            REACTION_TRIALS = 100
            REACTION_BLOCKS = 3
            REACTION_TRIAL_TIME = 8000
            TASK_BLOCKS = 4
            TASK_TRIALS = 80
        TIMING_TRIALS = 50
        TRIAL_TIME = 2000
        
    #* Get pull list and num subjects
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
        NUM_SUBJECTS = len(figures_pull_list)
    
    #* Load control data
    if True:
        # ---------------Controls-------------------------
        reaction_trial_start          = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS, REACTION_TRIALS))*np.nan
        reaction_filenames            = np.empty((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS),dtype = object)
        agent_reaction_leave_time     = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS))*np.nan
        agent_reaction_decision_array = np.empty((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS))*np.nan
        reaction_trial_type_array     = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS))*np.nan
        reaction_xypos_data           = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS, REACTION_TRIAL_TIME,2))*np.nan
        reaction_dist_data            = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS, REACTION_TRIAL_TIME))*np.nan
        reaction_xyvelocity_data      = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS, REACTION_TRIAL_TIME,2))*np.nan
        reaction_speed_data           = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS, REACTION_TRIAL_TIME))*np.nan
        # reaction_xyforce_data         = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS, REACTION_TRIAL_TIME,2))*np.nan
        # reaction_force_data           = np.zeros((NUM_SUBJECTS, REACTION_BLOCKS,REACTION_TRIALS, REACTION_TRIAL_TIME))*np.nan 

        coincidence_trial_start                       = np.zeros((NUM_SUBJECTS, COINCIDENCE_TRIALS))*np.nan
        coincidence_reach_time                        = np.zeros((NUM_SUBJECTS, COINCIDENCE_TRIALS))*np.nan
        interval_trial_start                          = np.zeros((NUM_SUBJECTS, INTERVAL_TRIALS))*np.nan
        interval_reach_time                           = np.zeros((NUM_SUBJECTS, INTERVAL_TRIALS))*np.nan

        for i in range(NUM_SUBJECTS):
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
        # ---------------Controls-------------------------
        task_trial_start          = np.zeros((NUM_SUBJECTS, TASK_BLOCKS, TASK_TRIALS))*np.nan
        task_filenames            = np.empty((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS),dtype = object)
        agent_task_leave_time     = np.zeros((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS))*np.nan
        agent_task_decision_array = np.empty((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS))*np.nan
        task_xypos_data           = np.zeros((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS, TRIAL_TIME,2))*np.nan
        task_dist_data            = np.zeros((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS, TRIAL_TIME))*np.nan
        task_xyvelocity_data      = np.zeros((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS, TRIAL_TIME,2))*np.nan
        task_speed_data           = np.zeros((NUM_SUBJECTS, TASK_BLOCKS,TASK_TRIALS, TRIAL_TIME))*np.nan 
        
        for i in range(NUM_SUBJECTS):
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
        data_path = 'Subjects_Analyzed\\'        
        subject_object = SubjectBuilder(
            subjects= figures_pull_list,experiment = experiment, num_task_trials_initial = TASK_TRIALS, num_task_blocks = TASK_BLOCKS, 
            num_reaction_blocks = REACTION_BLOCKS, num_reaction_trials = REACTION_TRIALS,num_timing_trials = TIMING_TRIALS, select_trials=select_trials,
            
            reaction_trial_start = reaction_trial_start, reaction_xypos_data = reaction_xypos_data,
            reaction_dist_data = reaction_dist_data, reaction_xyvelocity_data = reaction_xyvelocity_data,
            reaction_speed_data = reaction_speed_data, reaction_trial_type_array = reaction_trial_type_array, 
            agent_reaction_decision_array = agent_reaction_decision_array,agent_reaction_leave_time = agent_reaction_leave_time, 
            
            task_xypos_data = task_xypos_data, task_dist_data = task_dist_data, task_xyvelocity_data = task_xyvelocity_data, task_speed_data = task_speed_data,
            interval_trial_start = interval_trial_start, interval_reach_time = interval_reach_time, coincidence_trial_start = coincidence_trial_start, coincidence_reach_time = coincidence_reach_time,
            agent_task_leave_time = agent_task_leave_time, agent_task_target_selection_array = agent_task_decision_array,
                                    )
    assert TRIAL_TIME == 2000    
    return subject_object