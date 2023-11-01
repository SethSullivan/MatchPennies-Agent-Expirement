import dill 
import numpy as np
import numba as nb

import read_data_functions as rdf
import constants

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
metrics_dict = dict(zip(keys,metrics))
with open(constants.MODEL_INPUT_PATH / 'model_input_dict.pkl','wb') as f:
    dill.dump(metrics_dict,f)
        
        
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

@nb.njit(parallel=True)
def parameter_bootstrap(parameters:np.ndarray,M=1e4,):
    num_params = len(parameters)
    distribution = np.empty((num_params,int(M)))
    results = np.empty((num_params))
    for i in nb.prange(num_params):
        for j in range(int(M)):
            distribution[i,j] = np.nanmean(np.random.choice(parameters[i], size=len(parameters[i]),replace=True))
        results[i] = np.mean(distribution[i,:])
    return distribution, results

# class BootstrapInputs:
#     def __init__(self,group,M=1e4):
#         self.rt = np.nanmedian(group.movement_metrics.reaction_times, axis=1) - 25
#         self.rt_sd = np.nanstd(group.movement_metrics.reaction_times, axis=1)
#         self.mt = np.min(np.nanmedian(group.movement_metrics.movement_times("task"), axis=2), axis=1)  # Get movement time for the condition where they tried the hardest
#         self.mt_sd = np.min(np.nanstd(group.movement_metrics.movement_times("task"), axis=2), axis=1)
#         self.timing_sd = np.nanstd(group.movement_metrics.coincidence_reach_time, axis=1) #! This needs to be shape = (6,)
#         self.agent_sds = np.nanstd(group.raw_data.agent_task_leave_time, axis=2)
#         self.agent_means = np.nanmean(group.raw_data.agent_task_leave_time, axis=2)
#         self.parameters = np.array([self.rt,self.rt_sd,self.mt,self.mt_sd,self.timing_sd])
        
#     @property
#     def get_boot_inputs(self):
#         self.boot_dist,self.boot_results = parameter_bootstrap(self.parameters,M=M)
#         keys = ['rt','rt_sd','mt','mt_sd','timing_sd','agent_mean','agent_sd']
#         results_dict = dict(zip(results_keys,results))
#         distribution_dict = dict(zip(results_keys,distribution))

# class ModelInputs:
#     def __init__(self,group,GROUP_OR_INDIVIDUAL):
#         if GROUP_OR_INDIVIDUAL.lower() == "group":
#             subnames = ['group']
#             with open(MODELS_PATH / "boostrapped_parameters.pkl",'rb') as f:
#                 inputs = dill.load(f)
#             with open(MODELS_PATH / "agent_sds.pkl",'rb') as f:
#                 agent_sds = dill.load(f) 
#             with open(MODELS_PATH / "agent_means.pkl",'rb') as f:
#                 agent_means = dill.load(open(MODELS_PATH / "agent_means",'rb'))
            
#             # Unpack inputs and put into list so they work in a loop
#             rt = [inputs['rt']]
#             rt_sd = [inputs['rt_sd']]
#             mt = [inputs['mt']]
#             mt_sd = [inputs['mt_sd']]
#             timing_sd = [inputs['timing_sd']]
            
#         else GROUP_OR_INDIVIDUAL.lower() == "individual":
#             subnames = [f'sub{i}' for i in range(it.num_subjects)]
#             rt = np.nanmedian(group.movement_metrics.reaction_times, axis=1) - 25
#             rt_sd = np.nanstd(group.movement_metrics.reaction_times, axis=1)

#             mt = np.min(np.nanmedian(group.movement_metrics.movement_times("task"), axis=2), axis=1)  # Get movement time for the condition where they tried the hardest
#             mt_sd = np.min(np.nanstd(group.movement_metrics.movement_times("task"), axis=2),axis = 1)
#             timing_sd = np.array([np.nanstd(group.movement_metrics.coincidence_reach_time, axis=1)]*it.num_blocks).T #! This needs to be shape = (6,)
#             guess_sd = np.nanstd(group.react_guess_movement_metrics.movement_onset_times("guess"), axis=2) # 10/25/23 UNUSED, switch sd and timing sd should soak this up, but it was low in the past
#             agent_sds = np.nanstd(group.raw_data.agent_task_leave_time, axis=2)
#             agent_means = np.nanmean(group.raw_data.agent_task_leave_time, axis=2)
#             guess_leave_time_sd = np.nanstd(group.react_guess_movement_metrics.movement_onset_times("guess"), axis=2) # UNUSED