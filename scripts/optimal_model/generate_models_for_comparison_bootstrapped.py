import numpy as np
import dill
from copy import deepcopy
from pathlib import Path
import pandas as pd
from datetime import datetime
import itertools
from tqdm import tqdm
import numba as nb

import read_data_functions as rdf
from Optimal_Stopping_Object import ModelConstructor, ModelFitting
from initializer import InitialThangs
import loss_functions as lf
import constants

# * Functions
def get_loss(model, targets, drop_condition_num=None):
    model_metrics = [
        model.player_behavior.wtd_leave_time,
        model.player_behavior.wtd_leave_time_sd,
    ]
    predictions = [model.results.get_metric(metric, decision_type="optimal", metric_type="true") for metric in model_metrics]
    loss = lf.ape_loss(predictions, targets, drop_condition_num=drop_condition_num)

    return loss

def create_input_row_dict(model, loss, model_name,):
    input_row_dict = {"Model": model_name, "Loss": loss}
    input_row_dict.update(vars(model.inputs))
    input_row_dict.pop("timesteps")

    return input_row_dict

def map_reward_change(score: float, comparison_num: float) -> str:
    if score > comparison_num:
        score_change = "Greater"
    elif score < comparison_num:
        score_change = "Less"
    else:
        score_change = "Normal"
    return score_change

@nb.njit(parallel=True)
def parameter_bootstrap(parameters:np.ndarray,M=1e4,):
    print('begin bootstrap')
    M = int(M)
    num_params = len(parameters)
    num_subjects = len(parameters[0])
    distribution = np.empty((M,num_params))
    results = np.empty((num_params))
    participant_ids = np.empty((M,num_subjects),dtype=np.int32) 
    for i in range(M):
        # One set of participant ids, use across every paramter
        participant_ids[i,:] = np.random.randint(0,num_subjects,size=num_subjects)
        
        # Use those ids to get all parameters
        for j in range(num_params):
            distribution[i,j] = np.nanmean(parameters[j][participant_ids[i,:]])
            
        results[i] = np.mean(distribution[i,:])
    return distribution, results, participant_ids
    
    
# * Select experiment you'd like to run
EXPERIMENTS = ["Exp1"]

#* DECIDE TO FIT PARAMETERS OR NOT
FIT_PARAMETERS = True
SAVE = False
M = 100

input_keys = ["rt","rt_sd","mt","mt_sd","timing_sd",]

for EXPERIMENT in EXPERIMENTS:
    # * Initial Thangs
    # Get path and save path
    LOAD_PATH = Path(f"D:\OneDrive - University of Delaware - o365\Subject_Data\MatchPennies_Agent_{EXPERIMENT}")
    MODELS_PATH = Path(f"D:\\OneDrive - University of Delaware - o365\\Desktop\\MatchPennies-Agent-Expirement\\results\\models")
    it = InitialThangs(EXPERIMENT)

    # # * Get group data
    # print("Reading in data...")
    # if "group" not in locals():
    #     group = rdf.generate_subject_object_v3(EXPERIMENT, "All Trials")
    # else:
    #     if group.exp_info.experiment != EXPERIMENT:  # This means i changed experiment and need to run again
    #         group = rdf.generate_subject_object_v3(EXPERIMENT, "All Trials")
    # print(f"Starting up {EXPERIMENT}")

    # #* Participant median arrays
    # # * Set the parameters that change with each model
    # rt = np.nanmedian(group.movement_metrics.reaction_times, axis=1) - 25
    # rt_sd = np.nanstd(group.movement_metrics.reaction_times, axis=1)
    # mt = np.min(np.nanmedian(group.movement_metrics.movement_times("task"), axis=2), axis=1)
    # mt_sd = np.min(np.nanstd(group.movement_metrics.movement_times("task"), axis=2), axis=1)
    # timing_sd = np.nanstd(group.movement_metrics.coincidence_reach_time, axis=1) #! This nee
    # agent_sds = np.nanmedian(np.nanstd(group.raw_data.agent_task_leave_time, axis=2),axis=0)
    # agent_means = np.nanmedian(np.nanmean(group.raw_data.agent_task_leave_time, axis=2),axis=0)
    with open(constants.MODELS_PATH / 'model_input_dict.pkl','rb') as f:
        model_input_dict = dill.load(f)
        
    bootstrap_parameters = np.array([x for x in model_input_dict.values() if x.shape[0]==it.num_subjects])

    distribution,results,participant_ids = parameter_bootstrap(parameters=bootstrap_parameters, M=M)
    
    print('done')
    with open(constants.MODELS_PATH / 'participant_median_movement_onset_time.pkl','rb') as f:
        participant_median_movement_onset_time = dill.load(f)
    with open(constants.MODELS_PATH / 'participant_sd_movement_onset_time.pkl','rb') as f:
        participant_sd_movement_onset_time = dill.load(f)
    
    for i in range(M):
        player_inputs = dict(zip(input_keys,distribution[i,:]))

        GUESS_SWITCH_DELAY = 65
        GUESS_SWITCH_SD = 65
        INCORRECT_CHANGE = -0.1
        INDECISION_CHANGE = 0
        
        params_dict = {
            "timing_sd_true": [[player_inputs['timing_sd']]*it.num_blocks],
            "timing_sd_expected": [[player_inputs['timing_sd']]*it.num_blocks, np.array([1]*it.num_blocks)],
            "guess_switch_delay_true": [GUESS_SWITCH_DELAY], #! Assuming guess switch delay always exists
            "guess_switch_delay_expected": [GUESS_SWITCH_DELAY, 1],
            "guess_switch_sd_true": [GUESS_SWITCH_SD], #! Assuming guess switch sd always exists
            "guess_switch_sd_expected": [GUESS_SWITCH_SD, 1],
        }
        
        #* Get all param combos 
        #! Not doing every combo, just fitting all of them and letting the optimizer say what's expected and what's true
        # all_param_combos = itertools.product(*params_dict.values())  # The * unpacks into the product function
        # * Get targets for model comparisons
        # comparison_targets = np.array(
        #     [
        #     np.nanmedian(np.nanmedian(group.movement_metrics.movement_onset_times("task"), axis=2), axis=0),
        #     np.nanmedian(group.score_metrics.score_metric("wins"), axis=0) / 100,
        #     np.nanmedian(group.score_metrics.score_metric("incorrects"), axis=0) / 100,
        #     np.nanmedian(group.score_metrics.score_metric("indecisions"), axis=0) / 100,
        #     ]   
        # ) 
        # Comparing to only the participant ids that are use

        # * Loop through all the changing parameters
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
            "Known_Timing_SD": params['timing_sd_expected'] == params['timing_sd_true'],
            }
            descriptive_parameters.append(descriptive_parameter_row)
            
            model = ModelConstructor(
                experiment=EXPERIMENT,
                num_blocks=it.num_blocks,
                num_timesteps=1800,
                agent_means=np.array([model_input_dict["agent_means"], model_input_dict["agent_means"]])[:, :, np.newaxis],
                agent_sds=np.array([model_input_dict["agent_sds"], model_input_dict["agent_sds"]])[:, :, np.newaxis],  #!
                reaction_time=np.array([player_inputs["rt"], player_inputs["rt"]])[:, np.newaxis, np.newaxis],
                movement_time=np.array([player_inputs["mt"], player_inputs["mt"]])[:, np.newaxis, np.newaxis],
                reaction_sd=np.array([player_inputs["rt_sd"], player_inputs["rt_sd"]])[:, np.newaxis, np.newaxis],  #! Reducing these, aka the particiapnt thinks they are more certain than they are
                movement_sd=np.array([player_inputs["mt_sd"], player_inputs["mt_sd"]])[:, np.newaxis, np.newaxis],
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
                comparison_targets = np.array(
                    [
                        np.nanmedian(participant_median_movement_onset_time[participant_ids[i,:]], axis=0),
                        np.nanmedian(participant_sd_movement_onset_time[participant_ids[i,:]], axis=0),
                    ]   
                )      
                metric_keys = ['wtd_leave_time','wtd_leave_time_sd']
                model_fit = ModelFitting(model=model)
                res = model_fit.run_model_fit_procedure(
                    free_params_init=free_params_init_with_sd,
                    targets=comparison_targets,
                    drop_condition_from_loss=None,  # Drop 1200 50
                    limit_sd=False,
                    metric_keys=metric_keys,
                    bnds=None,
                    tol=0.000000001,
                    method="Powell",
                )
                if descriptive_parameter_row['Known_Switch_Delay'] and descriptive_parameter_row['Known_Switch_SD']:
                    assert np.all(model.inputs.guess_switch_delay[0] == model.inputs.guess_switch_delay[1])
                    
            loss = model_fit.loss_store[-1]
            descriptive_parameter_row['Loss'] = loss
            input_row_dict = create_input_row_dict(model, loss, model_name)
            input_parameters.append(input_row_dict)

            c += 1

        df_inputs = pd.DataFrame(input_parameters)
        df_descriptions = pd.DataFrame(descriptive_parameters)
        if SAVE:
            save_date = datetime.now()
            # * Save the old model table to a new file
            with open(MODELS_PATH / f"{EXPERIMENT}_model_parameters_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
                dill.dump(df_inputs, f)

            with open(MODELS_PATH / f"{EXPERIMENT}_model_descriptions_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
                dill.dump(df_descriptions, f)

        print(f"Model generation for {EXPERIMENT} completed")
