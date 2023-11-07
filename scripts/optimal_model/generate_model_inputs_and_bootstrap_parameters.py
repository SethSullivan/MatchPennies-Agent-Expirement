import dill 
import numpy as np
import numba as nb

import read_data_functions as rdf
import constants
from initializer import InitialThangs
from Optimal_Stopping_Object import ModelConstructor

it = InitialThangs("Exp1")
#* Pickle model inputs
group = rdf.generate_subject_object_v3('Exp1', "All Trials")
rt = np.nanmedian(group.movement_metrics.reaction_times, axis=1) - 25
rt_sd = np.nanstd(group.movement_metrics.reaction_times, axis=1)
mt = np.min(np.nanmedian(group.movement_metrics.movement_times("task"), axis=2), axis=1)
mt_sd = np.min(np.nanstd(group.movement_metrics.movement_times("task"), axis=2), axis=1)
timing_sd = np.nanstd(group.movement_metrics.coincidence_reach_time, axis=1) #! This nee
agent_sds = np.nanmedian(np.nanstd(group.raw_data.agent_task_leave_time, axis=2),axis=0)
agent_means = np.nanmedian(np.nanmean(group.raw_data.agent_task_leave_time, axis=2),axis=0)
metrics = [rt,rt_sd,mt,mt_sd,timing_sd,agent_sds,agent_means]
keys = ["rt","rt_sd","mt","mt_sd","timing_sd","agent_sds","agent_means"]
model_input_dict = dict(zip(keys,metrics))
with open(constants.MODEL_INPUT_PATH / 'model_input_dict.pkl','wb') as f:
    dill.dump(model_input_dict,f)

#* Pickle true median data for loss function
participant_median_movement_onset_time = np.nanmedian(group.movement_metrics.movement_onset_times("task"), axis=2)
with open(constants.MODEL_INPUT_PATH / 'participant_median_movement_onset_time.pkl','wb') as f:
    dill.dump(participant_median_movement_onset_time,f)
participant_sd_movement_onset_time = np.nanstd(group.movement_metrics.movement_onset_times("task"), axis=2)
with open(constants.MODEL_INPUT_PATH / 'participant_sd_movement_onset_time.pkl','wb') as f:
    dill.dump(participant_sd_movement_onset_time,f)
wins = group.score_metrics.score_metric("wins")
with open(constants.MODEL_INPUT_PATH / 'participant_wins.pkl','wb') as f:
    dill.dump(wins,f)
incorrects = group.score_metrics.score_metric("incorrects")
with open(constants.MODEL_INPUT_PATH / 'participant_incorrects.pkl','wb') as f:
    dill.dump(incorrects,f)
indecisions = group.score_metrics.score_metric("indecisions")
with open(constants.MODEL_INPUT_PATH / 'participant_indecisions.pkl','wb') as f:
    dill.dump(indecisions,f)


#* Bootstrap parameters
@nb.njit(parallel=True)
def parameter_bootstrap(parameters:np.ndarray,M=1e4,):
    print('begin bootstrap')
    M = int(M)
    num_params = len(parameters)
    num_subjects = len(parameters[0])
    distribution = np.zeros((M,num_params))*np.nan
    results = np.zeros((M,num_params))*np.nan
    participant_ids = np.zeros((M,num_subjects),dtype=np.int32)
    for i in nb.prange(M):
        # One set of participant ids, use across every paramter
        participant_ids[i,:] = np.random.randint(0,num_subjects,size=num_subjects)
        
        # Use those ids to get all parameters
        for j in range(num_params):
            distribution[i,j] = np.nanmean(parameters[j][participant_ids[i,:]])
            
        results[i,:] = np.mean(distribution[i,:])
        
    return distribution, results, participant_ids

bootstrap_parameters = np.array([x for x in model_input_dict.values() if x.shape[0]==it.num_subjects])
a,b,c = parameter_bootstrap(parameters=bootstrap_parameters, M=1e2) # initialize bootstrap

parameter_distribution,results,participant_ids = parameter_bootstrap(parameters=bootstrap_parameters, )
print("finished bootstrap")
with open(constants.MODEL_INPUT_PATH / 'bootstrap_parameter_distribution.pkl','wb') as f:
    dill.dump(parameter_distribution,f)    
with open(constants.MODEL_INPUT_PATH / 'bootstrap_results.pkl','wb') as f:
    dill.dump(results,f)    
with open(constants.MODEL_INPUT_PATH / 'participant_ids.pkl','wb') as f:
    dill.dump(participant_ids,f)  