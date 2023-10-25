import numpy as np
import dill
from pathlib import Path
import pandas as pd
import numba as nb
import read_data_functions as rdf
from initializer import InitialThangs

from tests.test_model import SAVE_PATH

@nb.njit(parallel=True)
def parameter_bootstrap(parameters:np.ndarray,M=1e5,):
    num_params = len(parameters)
    distribution = np.empty((num_params,int(M)))
    results = np.empty((num_params))
    for i in nb.prange(num_params):
        for j in range(int(M)):
            distribution[i,j] = np.nanmean(np.random.choice(parameters[i], size=len(parameters[i]),replace=True))
        results[i] = np.mean(distribution[i,:])
    return distribution, results
        
#* !! SELECT THESE BEFORE RUNNING !!
EXPERIMENT = "Exp1"
FIT_PARAMETERS = False

# * GET THE MODEL TRACKER TABLE
SAVE_PATH = Path(f"D:\\OneDrive - University of Delaware - o365\\Desktop\\MatchPennies-Agent-Expirement\\results")

# * Initial Thangs
# Get path and save path
LOAD_PATH = Path(f"D:\OneDrive - University of Delaware - o365\Subject_Data\MatchPennies_Agent_{EXPERIMENT}")
it = InitialThangs(EXPERIMENT)

# * Get group data
if "group" not in locals():
    group = rdf.generate_subject_object_v3(EXPERIMENT, "All Trials")
else:
    if group.exp_info.experiment != EXPERIMENT:  # This means i changed experiment and need to run again
        group = rdf.generate_subject_object_v3(EXPERIMENT, "All Trials")

#* Get raw input data
    
rt = np.nanmedian(group.movement_metrics.reaction_times, axis=1) - 25
rt_sd = np.nanstd(group.movement_metrics.reaction_times, axis=1)
mt = np.min(np.nanmedian(group.movement_metrics.movement_times("task"), axis=2), axis=1)  # Get movement time for the condition where they tried the hardest
mt_sd = np.min(np.nanstd(group.movement_metrics.movement_times("task"), axis=2), axis=1)
timing_sd = np.nanstd(group.movement_metrics.coincidence_reach_time, axis=1) #! This needs to be shape = (6,)

parameters = np.array([rt,rt_sd,mt,mt_sd,timing_sd])

distribution, results = parameter_bootstrap(parameters, M=1e6)
results_keys = ['rt','rt_sd','mt','mt_sd','timing_sd']
results_dict = dict(zip(results_keys,results))
distribution_dict = dict(zip(results_keys,distribution))

#* Don't think I need to bootstrap these
agent_sds = np.nanstd(group.raw_data.agent_task_leave_time, axis=2)
agent_means = np.nanmean(group.raw_data.agent_task_leave_time, axis=2)

save_objs = [results_dict,distribution_dict,agent_sds,agent_means]
save_names = ["bootstrapped_parameters","boostrapped_parameter_distributions",
              "agent_sds","agent_means"]
for obj,save_name in zip(save_objs,save_names):
    with open(SAVE_PATH / f"{save_name}.pkl","wb") as f:
        dill.dump(obj,f)
