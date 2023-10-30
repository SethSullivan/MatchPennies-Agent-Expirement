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

# @nb.njit(parallel=True)
def parameter_bootstrap(parameters:np.ndarray,M=1e4,):
    print('begin bootstrap')
    M = int(M)
    num_params = len(parameters)
    num_subjects = len(parameters[0])
    distribution = np.zeros((M,num_params))*np.nan
    results = np.zeros((M,num_params))*np.nan
    participant_ids = np.zeros((M,num_subjects),dtype=np.int32)
    for i in range(M):
        # One set of participant ids, use across every paramter
        participant_ids[i,:] = np.random.randint(0,num_subjects,size=num_subjects)
        
        # Use those ids to get all parameters
        for j in range(num_params):
            distribution[i,j] = np.nanmean(parameters[j][participant_ids[i,:]])
            
        results[i,:] = np.mean(distribution,axis=0)
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

    parameter_distribution,results,participant_ids = parameter_bootstrap(parameters=bootstrap_parameters, M=M)
    
    print('done')
    with open(constants.MODELS_PATH / 'participant_median_movement_onset_time.pkl','rb') as f:
        participant_median_movement_onset_time = dill.load(f)
    with open(constants.MODELS_PATH / 'participant_sd_movement_onset_time.pkl','rb') as f:
        participant_sd_movement_onset_time = dill.load(f)
            
    input_parameters = []
    print("Starting Models...")
    for i in range(1):
        player_inputs = dict(zip(input_keys,parameter_distribution[i,:]))

        # * Loop through all the changing parameters
        model_name = f"model{i}_{datetime.now():%Y_%m_%d_%H_%M_%S}"

        #* 3 Models 
        #* 1. Full optimal, no fitting, no switch delay or uncertainty
        #* 2. Full optimal, accounting for fit switch delay and uncertainty, and the expected and true have to be equal
        #* 3. Full optimal, not accounting for fit switch delay and uncertainty, and the expected and true are both fit simultaneously
        
        optimal_model_no_switch = ModelConstructor(
            experiment=EXPERIMENT,
            num_blocks=it.num_blocks,
            num_timesteps=1800,
            agent_means=np.array([model_input_dict["agent_means"], model_input_dict["agent_means"]])[:, :, np.newaxis],
            agent_sds=np.array([model_input_dict["agent_sds"], model_input_dict["agent_sds"]])[:, :, np.newaxis],  #!
            reaction_time=np.array([player_inputs["rt"], player_inputs["rt"]])[:, np.newaxis, np.newaxis],
            movement_time=np.array([player_inputs["mt"], player_inputs["mt"]])[:, np.newaxis, np.newaxis],
            reaction_sd=np.array([player_inputs["rt_sd"], player_inputs["rt_sd"]])[:, np.newaxis, np.newaxis],  #! Reducing these, aka the particiapnt thinks they are more certain than they are
            movement_sd=np.array([player_inputs["mt_sd"], player_inputs["mt_sd"]])[:, np.newaxis, np.newaxis],
            timing_sd=np.array([[player_inputs['timing_sd']]*it.num_blocks, 
                                [player_inputs['timing_sd']]*it.num_blocks])[:, :, np.newaxis],
            guess_switch_delay=np.array([0, 0])[:, np.newaxis, np.newaxis], # These are being FIT, so copied models can just have them as 0
            guess_switch_sd=np.array([0,0])[:, np.newaxis, np.newaxis],   
            electromechanical_delay=np.array([50, 50])[:, np.newaxis, np.newaxis],
            electromechanical_sd = np.array([10,10])[:, np.newaxis, np.newaxis],
            switch_cost_exists=True,
            expected=False,  
            win_reward=1.0,
            incorrect_cost=0.0,  #! These are applied onto the base reward matrix in Optimal Model object
            indecision_cost=0.0,
            round_num = 20,
        )
        #* Switch true will get fit
        optimal_model_with_switch = deepcopy(optimal_model_no_switch)
        
        suboptimal_model_with_switch = ModelConstructor(
            experiment=EXPERIMENT,
            num_blocks=it.num_blocks,
            num_timesteps=1800,
            agent_means=np.array([model_input_dict["agent_means"], model_input_dict["agent_means"]])[:, :, np.newaxis],
            agent_sds=np.array([model_input_dict["agent_sds"], model_input_dict["agent_sds"]])[:, :, np.newaxis],  #!
            reaction_time=np.array([player_inputs["rt"], player_inputs["rt"]])[:, np.newaxis, np.newaxis],
            movement_time=np.array([player_inputs["mt"], player_inputs["mt"]])[:, np.newaxis, np.newaxis],
            reaction_sd=np.array([player_inputs["rt_sd"], player_inputs["rt_sd"]])[:, np.newaxis, np.newaxis],  #! Reducing these, aka the particiapnt thinks they are more certain than they are
            movement_sd=np.array([player_inputs["mt_sd"], player_inputs["mt_sd"]])[:, np.newaxis, np.newaxis],
            timing_sd=np.array([[player_inputs['timing_sd']]*it.num_blocks, 
                                [player_inputs['timing_sd']]*it.num_blocks])[:, :, np.newaxis],
            guess_switch_delay=np.array([0, 0])[:, np.newaxis, np.newaxis], # These are being FIT, so copied models can just have them as 0
            guess_switch_sd=np.array([0,0])[:, np.newaxis, np.newaxis],   
            electromechanical_delay=np.array([50, 50])[:, np.newaxis, np.newaxis],
            electromechanical_sd = np.array([10,10])[:, np.newaxis, np.newaxis],
            switch_cost_exists=True,
            expected=True, 
            win_reward=1.0,
            incorrect_cost=0.0,  #! These are applied onto the base reward matrix in Optimal Model object
            indecision_cost=0.0,
            round_num = 20,
        )
        #* Switch true and expected will get fit
        optimal_model_no_switch.run_model()
        optimal_model_with_switch.run_model()
        suboptimal_model_with_switch.run_model()

        if FIT_PARAMETERS:
            #* We fit just the true, don't need the expected because we don't use those for optimal model get_metrics()
            free_params_optimal = {
                "guess_switch_delay_true": 0,
                "guess_switch_sd_true": 0,
                }
            
            #* Fit the true and expected separately and see what the model does
            free_params_suboptimal = {
                "guess_switch_delay_true": 0,
                "guess_switch_delay_expected": 0,
                "guess_switch_sd_true": 0,
                "guess_switch_sd_expected": 0,
                }
            # Need to be in for loop because we're using specific participant_ids
            comparison_targets = np.array(
                [
                    np.nanmedian(participant_median_movement_onset_time[participant_ids[i,:]], axis=0),
                    np.nanmedian(participant_sd_movement_onset_time[participant_ids[i,:]], axis=0),
                ]   
            )      
            metric_keys = ['wtd_leave_time','wtd_leave_time_sd']
            models = [optimal_model_with_switch,suboptimal_model_with_switch]
            free_params = [free_params_optimal, free_params_suboptimal]
            specific_name = ['optimal_','suboptimal_']
            model_fit_objects= []
            model_fit_results = []
            for j,model in enumerate(models):
                print('Fitting Model')
                print(model)
                model_fit = ModelFitting(model=model)
                print(model_fit)
                res = model_fit.run_model_fit_procedure(
                    free_params_init=free_params[i],
                    targets=comparison_targets,
                    drop_condition_from_loss=None,  # Drop 1200 50
                    limit_sd=False,
                    metric_keys=metric_keys,
                    bnds=None,
                    tol=0.000000001,
                    method="Powell",
                )
                specific_model_name = specific_name[j] + model_name
                model_fit_objects.append(model_fit)
                model_fit_results.append(res)
                loss = model_fit.loss_store[-1]
                print(loss)
                input_row_dict = create_input_row_dict(model, loss, specific_model_name)
                input_parameters.append(input_row_dict)

        df_inputs = pd.DataFrame(input_parameters)
        if SAVE:
            save_date = datetime.now()
            # * Save the old model table to a new file
            with open(MODELS_PATH / f"{EXPERIMENT}_three_model_comparison_model_parameters_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
                dill.dump(df_inputs, f)


        print(f"Model generation for {EXPERIMENT} completed")
