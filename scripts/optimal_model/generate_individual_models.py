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

#* !! SELECT THESE BEFORE RUNNING !!
EXPERIMENT = "Exp1"
FIT_PARAMETERS = False
GROUP = True # Run group model or individual models

# * GET THE MODEL TRACKER TABLE
MODELS_PATH = Path(f"D:\\OneDrive - University of Delaware - o365\\Desktop\\MatchPennies-Agent-Expirement\\results\\models")

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




for i in range(it.num_subjects):
    # * Set the parameters that change with each model
    if True:
        GUESS_SWITCH_DELAY = 65
        GUESS_SWITCH_SD = 65

        params_dict = {
            "timing_sd_true": [timing_sd[i]],
            "timing_sd_expected": [timing_sd[i], np.array([1]*it.num_blocks)],
            "guess_switch_delay_true": [GUESS_SWITCH_DELAY], #! Assuming guess switch delay always exists
            "guess_switch_delay_expected": [GUESS_SWITCH_DELAY, 1],
            "guess_switch_sd_true": [GUESS_SWITCH_SD], #! Assuming guess switch sd always exists
            "guess_switch_sd_expected": [GUESS_SWITCH_SD, 1],
        }
        # * Get targets for model comparisons
        comparison_targets = np.array(
            [         
                np.nanmedian(group.movement_metrics.movement_onset_times("task"), axis=2)[i],
                group.score_metrics.score_metric("wins")[i] / 100,
                group.score_metrics.score_metric("incorrects")[i] / 100,
                group.score_metrics.score_metric("indecisions")[i] / 100,
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
        model_name = f"sub{i}_model{c}_{datetime.now():%Y_%m_%d_%H_%M_%S}"
 
        descriptive_parameter_row = {
            "Model": model_name,
            "Loss": 0,
            "Known_Switch_Delay": params['guess_switch_delay_expected'] == params['guess_switch_delay_true'],
            "Known_Switch_SD": params['guess_switch_sd_expected'] == params['guess_switch_sd_true'],
            "Known_Timing_SD": params['timing_sd_expected'] == params['timing_sd_true'],
        }
        descriptive_parameters.append(descriptive_parameter_row)
        
        model = ModelConstructor(
            experiment=EXPERIMENT,
            num_blocks=it.num_blocks,
            num_timesteps=1800,
            agent_means=np.array([agent_means[i], agent_means[i]])[:, :, np.newaxis],
            agent_sds=np.array([agent_sds[i], agent_sds[i]])[:, :, np.newaxis],  #!
            reaction_time=np.array([rt[i], rt[i]])[:, np.newaxis, np.newaxis],
            movement_time=np.array([mt[i], mt[i]])[:, np.newaxis, np.newaxis],
            reaction_sd=np.array([rt_sd[i], rt_sd[i]])[:, np.newaxis, np.newaxis],  #! Reducing these, aka the particiapnt thinks they are more certain than they are
            movement_sd=np.array([mt_sd[i], mt_sd[i]])[:, np.newaxis, np.newaxis],
            timing_sd=np.array([params["timing_sd_true"], 
                                params["timing_sd_expected"]])[:, :, np.newaxis],
            guess_switch_delay=np.array([params["guess_switch_delay_true"], 
                                         params["guess_switch_delay_expected"]])[:, np.newaxis, np.newaxis],  # Designed like this for broadcasting reasons
            guess_switch_sd=np.array([params["guess_switch_sd_true"], 
                                      params["guess_switch_sd_expected"]])[:, np.newaxis, np.newaxis],  # This includes electromechanical delay sd and timing sd bc it's straight from data
            # guess_sd =  np.array([params['guess_sd_true'],params['guess_sd_expected']])[:,:,np.newaxis], # This includes electromechanical delay sd
            electromechanical_delay=np.array([50, 50])[:, np.newaxis, np.newaxis],
            switch_cost_exists=True,
            expected=True,  #! Should always be True... if the parameter is ground truth, then the two values of the parameter array should be the same
            win_reward=1.0,
            incorrect_cost=0.0,  #! These are applied onto the base reward matrix in Optimal Model object
            indecision_cost=0.0,
            round_num = 20,
        )
        
        if FIT_PARAMETERS:
            # If both known, then don't use true and expected
            if descriptive_parameter_row['Known_Switch_Delay'] and descriptive_parameter_row['Known_Switch_SD']:
                free_params_init_with_sd = {
                    "guess_switch_delay": 0,
                    "guess_switch_sd": 0,
                }
            # Separate exp and true for switch sd
            elif descriptive_parameter_row['Known_Switch_Delay'] and not descriptive_parameter_row['Known_Switch_SD']:
                free_params_init_with_sd = {
                    "guess_switch_delay": 0,
                    "guess_switch_sd_true": 0,
                    "guess_switch_sd_expected": 0,
                }
            elif not descriptive_parameter_row['Known_Switch_Delay'] and descriptive_parameter_row['Known_Switch_SD']:
                free_params_init_with_sd = {
                    "guess_switch_delay_true": 0,
                    "guess_switch_delay_expected": 0,
                    "guess_switch_sd": 0,
                }
            else:
                free_params_init_with_sd = {
                    "guess_switch_delay_true": 0,
                    "guess_switch_delay_expected": 0,
                    "guess_switch_sd_true": 0,
                    "guess_switch_sd_expected": 0,
                }
                
            behavior_targets_with_sd = np.array(
                [
                    np.nanmedian(group.movement_metrics.movement_onset_times('task'), axis=2)[i],
                    np.nanstd(group.movement_metrics.movement_onset_times('task'), axis=2)[i]
                ]
            )
            behavior_metric_keys_with_sd = ["wtd_leave_time","wtd_leave_time_sd",]
            model_fit_object_known = ModelFitting(model=model)
            res = model_fit_object_known.run_model_fit_procedure(
                free_params_init=free_params_init_with_sd,
                targets=behavior_targets_with_sd,
                drop_condition_from_loss=None, 
                limit_sd=False,
                metric_keys=behavior_metric_keys_with_sd,
                bnds=None,
                tol=0.000000001,
                method="Powell",
            )
            if descriptive_parameter_row['Known_Switch_Delay'] and descriptive_parameter_row['Known_Switch_SD']:
                assert np.all(model.inputs.guess_switch_delay[0] == model.inputs.guess_switch_delay[1])
                
        loss = get_loss(
            model,
            comparison_targets,
        )
        descriptive_parameter_row['Loss'] = loss
        input_row_dict = create_input_row_dict(model, loss, model_name)
        input_parameters.append(input_row_dict)

        c += 1

    df_inputs = pd.DataFrame(input_parameters)
    df_descriptions = pd.DataFrame(descriptive_parameters)

    save_date = datetime.now()
    # * Save the old model table to a new file
    with open(MODELS_PATH / f"{EXPERIMENT}_sub{i}_model_parameters_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
        dill.dump(df_inputs, f)

    with open(MODELS_PATH / f"{EXPERIMENT}_sub{i}_model_descriptions_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
        dill.dump(df_descriptions, f)

    print(f"Model generation for {EXPERIMENT}, Sub{i} completed")