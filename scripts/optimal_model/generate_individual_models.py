import numpy as np
import dill
from copy import deepcopy
from pathlib import Path
import pandas as pd
from datetime import datetime
import itertools
from tqdm import tqdm

import read_data_functions as rdf
from Optimal_Stopping_Object import ModelConstructor, ModelFitting
from initializer import InitialThangs
import loss_functions as lf
'''
Assumptions
1. Reaction time gets 25 subtracted off
  - This fits the best when the movement onset times are selected for the average model
  - Justify bc people try harder when reward is on the table
2. Movement time mean is the median movement time from the condition where participants tried their hardest (aka min of all the conditions)
3. We fit the free parameters for everyone in the 
4. 

What I'm doing
1. Fit the true and expected gamble switch and gamble delay for every person
  - 

'''
# * Functions
def get_loss(model, targets, drop_condition_num=None):
    model_metrics = [
        model.player_behavior.wtd_leave_time,
        model.score_metrics.prob_win,
        model.score_metrics.prob_incorrect,
        model.score_metrics.prob_indecision,
    ]
    predictions = [model.results.get_metric(metric, decision_type="optimal", metric_type="true") for metric in model_metrics]
    loss = lf.ape_loss(predictions, targets, drop_condition_num=drop_condition_num)

    return loss

def create_input_row_dict(model, loss, model_name,):
    input_row_dict = {"Model": model_name, "Loss": loss}
    input_row_dict.update(vars(model.inputs))
    input_row_dict.pop("timesteps")

    return input_row_dict

EXPERIMENT = "Exp1"

# * GET THE MODEL TRACKER TABLE
MODELS_PATH = Path(f"D:\\OneDrive - University of Delaware - o365\\Desktop\\MatchPennies-Agent-Expirement\\results\\{EXPERIMENT}\models")

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

if True:
    if EXPERIMENT == "Exp1":
        rt = np.nanmedian(group.movement_metrics.reaction_times, axis=1) - 25
        rt_sd = np.nanstd(group.movement_metrics.reaction_times, axis=1)

    mt = np.min(np.nanmedian(group.movement_metrics.movement_times("task"), axis=2), axis=0)  # Get movement time for the condition where they tried the hardest
    mt_sd = np.nanstd(group.movement_metrics.movement_times("task"), axis=1)
    time_sd = np.array([(np.nanstd(group.movement_metrics.coincidence_reach_time, axis=1))] * it.num_blocks)
    guess_sd = np.nanstd(group.react_guess_movement_metrics.movement_onset_times("guess"), axis=2)
    agent_sds = np.nanstd(group.raw_data.agent_task_leave_time, axis=2)
    agent_means = np.nanmean(group.raw_data.agent_task_leave_time, axis=2)
    guess_leave_time_sd = np.nanstd(group.react_guess_movement_metrics.movement_onset_times("guess"), axis=2)
    
for i in range(it.num_subjects):
    # * Set the parameters that change with each model
    if True:
        GUESS_SWITCH_DELAY = 65
        GUESS_SWITCH_SD = 65

        params_dict = {
            "timing_sd_change": [time_sd[0] / 2, 0],
            "guess_switch_delay_true": [GUESS_SWITCH_DELAY], #! Assuming guess switch delay always exists
            "guess_switch_delay_expected": [GUESS_SWITCH_DELAY, 1],
            "guess_switch_sd_true": [GUESS_SWITCH_SD], #! Assuming guess switch sd always exists
            "guess_switch_sd_expected": [GUESS_SWITCH_SD, 1],
        }
        # * Get targets for model comparisons
        comparison_targets = np.array(
            [         
                np.nanmedian(np.nanmedian(group.movement_metrics.movement_onset_times("task"), axis=2), axis=0),
                np.nanmedian(group.score_metrics.score_metric("wins"), axis=0) / 100,
                np.nanmedian(group.score_metrics.score_metric("incorrects"), axis=0) / 100,
                np.nanmedian(group.score_metrics.score_metric("indecisions"), axis=0) / 100,
            ]
        )
        #* Get all param combos
        all_param_combos = itertools.product(*params_dict.values())  # The * unpacks into the product function
        
    #* Loop through changing parameters
    c = 0
    input_parameters = []
    descriptive_parameters = []  # Used for saying what changed, as opposed to the actual parameter values
    param_keys = params_dict.keys()
    print("Starting Models...")
    for param_tuple in tqdm(all_param_combos):
        params = dict(zip(param_keys, param_tuple))
        model_name = f"model{c}_{datetime.now():%Y_%m_%d_%H_%M_%S}"

        descriptive_parameter_row = {
            "Model": model_name,
            "Loss": 0,
            "Known_Switch_Delay": params['guess_switch_delay_expected'] == params['guess_switch_delay_true'],
            "Known_Switch_SD": params['guess_switch_sd_expected'] == params['guess_switch_sd_true'],
            "Known_Timing_SD": params["timing_sd_change"] == 0,
        }
        descriptive_parameters.append(descriptive_parameter_row)
        
        model = ModelConstructor(
            experiment=EXPERIMENT,
            num_blocks=it.num_blocks,
            num_timesteps=1800,
            agent_means=np.array([agent_means, agent_means])[:, :, np.newaxis],
            agent_sds=np.array([agent_sds, agent_sds + params.get("agent_sd_change",0)])[:, :, np.newaxis],  #!
            reaction_time=np.array([rt, rt])[:, np.newaxis, np.newaxis],
            movement_time=np.array([mt, mt])[:, np.newaxis, np.newaxis],
            reaction_sd=np.array([rt_sd, rt_sd - params.get("rt_sd_change",0)])[
                :, np.newaxis, np.newaxis
            ],  #! Reducing these, aka the particiapnt thinks they are more certain than they are
            movement_sd=np.array([mt_sd, mt_sd - params.get("mt_sd_change",0)])[:, np.newaxis, np.newaxis],
            timing_sd=np.array([time_sd, time_sd - params["timing_sd_change"]])[:, :, np.newaxis],
            guess_switch_delay=np.array([params["guess_switch_delay_true"], 
                                         params["guess_switch_delay_expected"]])[:, np.newaxis, np.newaxis],  # Designed like this for broadcasting reasons
            guess_switch_sd=np.array([params["guess_switch_sd_true"], 
                                      params["guess_switch_sd_expected"]])[:, np.newaxis, np.newaxis],  # This includes electromechanical delay sd and timing sd bc it's straight from data
            # guess_sd =  np.array([params['guess_sd_true'],params['guess_sd_expected']])[:,:,np.newaxis], # This includes electromechanical delay sd
            electromechanical_delay=np.array([50, 50])[:, np.newaxis, np.newaxis],
            switch_cost_exists=True,
            expected=True,  #! Should always be True... if the parameter is ground truth, then the two values of the parameter array should be the same
            win_reward=params["score_rewards_list"][0],
            incorrect_cost=params["score_rewards_list"][1],  #! These are applied onto the base reward matrix in Optimal Model object
            indecision_cost=params["score_rewards_list"][2],
            round_num = 20,
        )