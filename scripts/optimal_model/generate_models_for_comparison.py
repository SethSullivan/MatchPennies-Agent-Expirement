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

def map_reward_change(score: float, comparison_num: float) -> str:
    if score > comparison_num:
        score_change = "Greater"
    elif score < comparison_num:
        score_change = "Less"
    else:
        score_change = "Normal"
    return score_change


# * Select experiment you'd like to run
EXPERIMENTS = ["Exp1"]

#* DECIDE TO FIT PARAMETERS OR NOT
FIT_PARAMETERS = True

SAVE = False

for EXPERIMENT in EXPERIMENTS:
    print(f'Starting up {EXPERIMENT}')
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

    # * Set inputs for models
    if True:
        if EXPERIMENT == "Exp1":
            rt = np.nanmedian(np.nanmedian(group.movement_metrics.reaction_times, axis=1)) - 25
            rt_sd = np.nanmedian(np.nanstd(group.movement_metrics.reaction_times, axis=1))

        elif EXPERIMENT == "Exp2":
            rt = np.nanmedian(np.nanmedian(group.movement_metrics.exp2_react_guess_reaction_time_split("react", "only"), axis=1)) - 25
            rt_sd = np.nanmedian(np.nanstd(group.movement_metrics.reaction_times, axis=2))

        mt = np.min(
            np.nanmedian(np.nanmedian(group.movement_metrics.movement_times("task"), axis=2), axis=0)
        )  # Get movement time for the condition where they tried the hardest
        mt_sd = np.nanmedian(np.nanstd(group.movement_metrics.movement_times("task"), axis=1))
        time_sd = np.array([np.nanmedian(np.nanstd(group.movement_metrics.coincidence_reach_time, axis=1))] * it.num_blocks)
        perc_wins_both_reach = np.nanmean(group.score_metrics.wins_when_both_reach(perc=True), axis=0)
        guess_sd = np.nanmedian(np.nanstd(group.react_guess_movement_metrics.movement_onset_times("guess"), axis=2), axis=0)
        agent_sds = np.nanmean(np.nanstd(group.raw_data.agent_task_leave_time, axis=2), axis=0)
        agent_means = np.nanmean(np.nanmean(group.raw_data.agent_task_leave_time, axis=2), axis=0)
        guess_leave_time_sd = np.nanmedian(np.nanstd(group.react_guess_movement_metrics.movement_onset_times("guess"), axis=2), axis=0)

    # * Set the parameters that change with each model
    if True:
        # Create keys of what's being changed
        model_dict_keys_sd = ["agent_sd", "rt_sd", "mt_sd", "timing_sd"]

        # (*LOOP 1*) List for altering each uncertainty
        agent_sd_change = [rt_sd / 2, 0]  # Cut it in half or keep it the same
        rt_sd_change = [rt_sd / 2, 0]
        mt_sd_change = [mt_sd / 2, 0]
        timing_sd_change = [time_sd[0] / 2, 0]

        GUESS_SWITCH_DELAY = 65
        GUESS_SWITCH_SD = 65
        INCORRECT_CHANGE = -0.1
        INDECISION_CHANGE = 0
        
        score_rewards_list = [[1.0, 0.0, 0.0]]#, [1.0, INCORRECT_CHANGE, INDECISION_CHANGE]]

        params_dict = {
            "agent_sd_change": [0],
            "timing_sd_change": [time_sd[0] / 2, 0],
            "guess_switch_delay_true": [GUESS_SWITCH_DELAY], #! Assuming guess switch delay always exists
            "guess_switch_delay_expected": [GUESS_SWITCH_DELAY, 1],
            "guess_switch_sd_true": [GUESS_SWITCH_SD], # ! Assuming guess switch sd always exists
            "guess_switch_sd_expected": [GUESS_SWITCH_SD, 1],
            "score_rewards_list": score_rewards_list,
            # "guess_sd_true":[guess_leave_time_sd],
            # "guess_sd_expected":[guess_leave_time_sd,guess_leave_time_sd/2],
        }

        # Option to remove parameters we don't care about
        PARAMS_TO_REMOVE = []
        if len(PARAMS_TO_REMOVE) != 0:
            for param in PARAMS_TO_REMOVE:
                params_dict.pop(param)
        
        #* Get all param combos
        all_param_combos = itertools.product(*params_dict.values())  # The * unpacks into the product function

    # * Get targets for model comparisons
    comparison_targets = np.array(
        [
            np.nanmedian(np.nanmedian(group.movement_metrics.movement_onset_times("task"), axis=2), axis=0),
            np.nanmedian(group.score_metrics.score_metric("wins"), axis=0) / 100,
            np.nanmedian(group.score_metrics.score_metric("incorrects"), axis=0) / 100,
            np.nanmedian(group.score_metrics.score_metric("indecisions"), axis=0) / 100,
        ]
    )
    # metric_keys = ['wtd_leave_time','prob_win','prob_incorrect','prob_indecision']

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
            # "Known_Guess_SD": np.all(model.inputs.guess_sd[0] == model.inputs.guess_sd[1]),
            "Known_Agent_SD": params["agent_sd_change"] == 0,
            "Known_RT_SD": params.get("rt_sd_change",0) == 0,
            "Known_MT_SD": params.get("mt_sd_change",0) == 0,
            "Known_Timing_SD": params["timing_sd_change"] == 0,
            "Win_Reward": map_reward_change(params["score_rewards_list"][0], comparison_num=1.0),
            "Incorrect_Cost": map_reward_change(params["score_rewards_list"][1], comparison_num=0.0),
            "Indecision_Cost": map_reward_change(params["score_rewards_list"][2], comparison_num=0.0),
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
        model.run_model()
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
                    np.nanmedian(np.nanmedian(group.movement_metrics.movement_onset_times('task'), axis=2), axis=0), 
                    np.nanmedian(np.nanstd(group.movement_metrics.movement_onset_times('task'), axis=2), axis=0),
                ]
            )
            behavior_metric_keys_with_sd = ["wtd_leave_time","wtd_leave_time_sd",]
            model_fit_object_known = ModelFitting(model=model)
            res = model_fit_object_known.run_model_fit_procedure(
                free_params_init=free_params_init_with_sd,
                targets=behavior_targets_with_sd,
                drop_condition_from_loss=None,  # Drop 1200 50
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
    if SAVE:
        save_date = datetime.now()
        # * Save the old model table to a new file
        with open(MODELS_PATH / f"{EXPERIMENT}_model_parameters_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
            dill.dump(df_inputs, f)

        with open(MODELS_PATH / f"{EXPERIMENT}_model_descriptions_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
            dill.dump(df_descriptions, f)

    print(f"Model generation for {EXPERIMENT} completed")
