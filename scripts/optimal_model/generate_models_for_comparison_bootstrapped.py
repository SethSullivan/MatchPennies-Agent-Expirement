import numpy as np
import dill
from copy import deepcopy
from pathlib import Path
import pandas as pd
from datetime import datetime
import itertools
from tqdm import tqdm
import numba as nb
import time
import read_data_functions as rdf
import model_helper_functions as mhf
from Optimal_Stopping_Object import ModelConstructor, ModelFitting
from initializer import InitialThangs
import loss_functions as lf
import constants


'''
This script can either do the warmstarts (set WARM_START = True)

It can also bootstrap using the best warmstart initial conditions
'''

# * Functions
def create_input_row_dict(model, loss, model_name,free_param_keys):
    input_row_dict = {"Model": model_name, 
                      "Loss": loss,
                      "fit_parameters":free_param_keys}
    input_row_dict.update(vars(model.inputs))
    input_row_dict.pop("timesteps")

    return input_row_dict

def create_results_row_dict(model,loss,model_name,free_param_keys):
    get_metric = model.results.get_metric
    model_data = [
        get_metric(model.player_behavior.wtd_leave_time,metric_type='true',decision_type='optimal'),
        get_metric(model.player_behavior.wtd_leave_time_sd,metric_type='true',decision_type='optimal'),
        get_metric(model.score_metrics.prob_indecision,metric_type='true',decision_type='optimal')*100,
        get_metric(model.score_metrics.prob_win,metric_type='true',decision_type='optimal')*100,
        get_metric(model.score_metrics.prob_incorrect,metric_type='true',decision_type='optimal')*100,
    ]
    results_row = {
        "Model":model_name,"Loss":loss,"fit_parameters":free_param_keys,
        "median_movement_onset_time":model_data[0],
        "sd_movement_onset_time":model_data[1],
        "indecisions":model_data[2],
        "wins":model_data[3],
        "incorrects":model_data[4],
    }
    return results_row

#* GLOBAL PARAMETERS
# Select experiment you'd like to run
EXPERIMENT = "Exp1"
it = InitialThangs(EXPERIMENT)

#! SET THE SETTINGS BEFORE RUNNING SCRIPT
print("DID YOU SET THE RIGHT SETTINGS?")
FIT_PARAMETERS = True
SAVE = True
MODEL_TO_FIT = "optimal"
WARM_START = False # If False, that means I'm bootstrapping with the warmstart initial condition 
STORE_BASE_MODEL = False
input_keys = ["rt","rt_sd","mt","mt_sd","timing_sd",]
print(f" Fit Parameters: {FIT_PARAMETERS}\n Save: {SAVE}\n Model to Fit: {MODEL_TO_FIT}\n Warm Start: {WARM_START}")
print(f" Fitting: {MODEL_TO_FIT}")
print(" ")
#* Load parameters, boostrap, and comparison targets
if True:
    #* True Parameters load
    with open(constants.MODEL_INPUT_PATH / 'model_input_dict.pkl','rb') as f:
        model_input_dict = dill.load(f)
    true_parameters = [np.nanmedian(v) for k,v in model_input_dict.items() if "agent" not in k] 
    #* Bootstrap Load
    with open(constants.MODEL_INPUT_PATH / 'bootstrap_parameter_distribution.pkl','rb') as f:
        parameter_distribution = dill.load(f)    
    with open(constants.MODEL_INPUT_PATH / 'bootstrap_results.pkl','rb') as f:
        results = dill.load(f)    
    with open(constants.MODEL_INPUT_PATH / 'participant_ids.pkl','rb') as f:
        participant_ids = dill.load(f)        
    #* Comparison Targets load
    with open(constants.MODEL_INPUT_PATH / 'participant_median_movement_onset_time.pkl','rb') as f:
        participant_median_movement_onset_time = dill.load(f)
    with open(constants.MODEL_INPUT_PATH / 'participant_sd_movement_onset_time.pkl','rb') as f:
        participant_sd_movement_onset_time = dill.load(f)
    with open(constants.MODEL_INPUT_PATH / 'participant_wins.pkl','rb') as f:
        participant_wins = dill.load(f)  
    with open(constants.MODEL_INPUT_PATH / 'participant_incorrects.pkl','rb') as f:
        participant_incorrects = dill.load(f)  
    with open(constants.MODEL_INPUT_PATH / 'participant_indecisions.pkl','rb') as f:
        participant_indecisions = dill.load(f)  
 
#* Run warm_start or boostrap using warmstart 
if WARM_START:
    print("FINDING INITIAL CONDITIONS")
    iters = 10000
    #* Randomize for warmstart
    player_inputs = dict(zip(input_keys,true_parameters)) #! This won't change unless we're boostrapping so can pull out of for loop for Warm_Start
    switch_delay_expected_rand = np.random.uniform(0,150,size=iters)
    switch_delay_true_rand = switch_delay_expected_rand + np.random.uniform(0,75,size=iters)
    switch_sd_expected_rand = np.random.uniform(0,200,size=iters)
    switch_sd_true_rand = switch_sd_expected_rand + np.random.uniform(0,75,size=iters)
    timing_sd_expected_rand = np.random.uniform(0,np.max(player_inputs['timing_sd']),size=iters)
else:
    print("BOOTSTRAPPING MODEL FITS USING WARMSTART")
    path = constants.MODELS_PATH / "warmstart_models"
    df_results = list(path.glob(f"{EXPERIMENT}_{MODEL_TO_FIT}_warmstart_results*"))[-1]
    df_inputs = list(path.glob(f"{EXPERIMENT}_{MODEL_TO_FIT}_warmstart_inputs*"))[-1]
    df_warmstart_results = pd.read_pickle(path / df_results)
    df_warmstart_inputs = pd.read_pickle(path / df_inputs)
    best_warmstart_inputs = df_warmstart_inputs[df_warmstart_inputs["Loss"] == df_warmstart_inputs["Loss"].min()].iloc[0]
    iters = participant_ids.shape[0]

print(f"ITERATIONS: {iters}")

initial_time = time.time()
base_input_parameters_for_df = []
base_results_for_df = []
input_parameters_for_df = []
results_for_df = []
print("Starting Models...")
for i in tqdm(range(iters)):
    if WARM_START:        
        # Optimal/Suboptimal handles the initial guess to free params mapping
        initial_guess = {
            "guess_switch_delay_true": switch_delay_true_rand[i],
            "guess_switch_delay_expected": switch_delay_expected_rand[i],
            "guess_switch_sd_true": switch_sd_true_rand[i],
            "guess_switch_sd_expected": switch_sd_expected_rand[i],
            "timing_sd_expected": timing_sd_expected_rand[i]
        }
    else: 
        # Player inputs are bootstrapped
        player_inputs = dict(zip(input_keys,parameter_distribution[i,:]))
        
        # Use initial guess dict from warmstart
        initial_guess = {
            "guess_switch_delay_true": best_warmstart_inputs["guess_switch_delay"].squeeze()[0],
            "guess_switch_delay_expected": best_warmstart_inputs["guess_switch_delay"].squeeze()[1],
            "guess_switch_sd_true": best_warmstart_inputs["guess_switch_sd"].squeeze()[0],
            "guess_switch_sd_expected": best_warmstart_inputs["guess_switch_sd"].squeeze()[1],
            "timing_sd_expected": best_warmstart_inputs["timing_sd"].squeeze()[1,0]
        }
    model_name = f"model{i}_{datetime.now():%Y_%m_%d_%H_%M_%S}"

    #* 3 Models 
    #* 1. Full optimal, no fitting, no switch delay or uncertainty
    #* 2. Full optimal, accounting for fit switch delay and uncertainty, and the expected and true have to be equal
    #* 3. Full optimal, not accounting for fit switch delay and uncertainty, and the expected and true are both fit simultaneously
    
    # Run pure optimal, no switch 
    # optimal_model_no_switch = mhf.run_model(player_inputs,
    #                                         expected=False,use_agent_behavior_lookup=False,
    #                                         round_num=20)

    # Run either optimal or suboptimal    
    if MODEL_TO_FIT == "optimal":
        fit_model = mhf.run_model(player_inputs,
                                  expected=False,use_agent_behavior_lookup=False,
                                  round_num=20)
        #! Not putting _true or _expected makes true == expected
        free_params = {
            "guess_switch_delay": initial_guess['guess_switch_delay_true'],
            "guess_switch_sd": initial_guess['guess_switch_sd_true'],
            }
        specific_name = 'optimal_'
        assert fit_model.inputs.expected == False
    elif MODEL_TO_FIT == "suboptimal":
        fit_model = mhf.run_model(player_inputs,
                                  expected=True,use_agent_behavior_lookup=False,
                                  round_num=20)
        assert fit_model.inputs.expected == True
        #* Fit the true and expected separately and see what the model does
        free_params = {
            "guess_switch_delay_true": initial_guess['guess_switch_delay_true'],
            "guess_switch_delay_expected": initial_guess['guess_switch_delay_expected'],
            "guess_switch_sd_true": initial_guess['guess_switch_sd_true'],
            "guess_switch_sd_expected": initial_guess['guess_switch_sd_expected'],
            "timing_sd_expected": initial_guess['timing_sd_expected']
            # "timing_sd_true":player_inputs["timing_sd"],
            # "timing_sd_expected":player_inputs["timing_sd"],
            # "reaction_time_true":player_inputs["rt"],
            # "reaction_time_expected":player_inputs["rt"],
            # "reaction_sd_true":player_inputs["rt_sd"],
            # "reaction_sd_expected":player_inputs["rt_sd"],
            # "movement_time_true":player_inputs["mt"],
            # "movement_time_expected":player_inputs["mt"],
            # "movement_sd_true":player_inputs["mt_sd"],
            # "movement_sd_expected":player_inputs["mt_sd"],
        }
        specific_name = 'suboptimal_'
        
    assert fit_model.inputs.round_num == 20     
    
    #! Need to be in for loop because we're using specific participant_ids if not warmstart
    if WARM_START:
        comparison_targets = np.array(
            [
                np.nanmedian(participant_median_movement_onset_time, axis=0),
                np.nanmedian(participant_sd_movement_onset_time, axis=0),
                np.nanmedian(participant_wins,axis=0)/it.num_trials,
                np.nanmedian(participant_incorrects,axis=0)/it.num_trials,
                np.nanmedian(participant_indecisions,axis=0)/it.num_trials,
            ]   
        )
    else:
        comparison_targets = np.array(
            [
                np.nanmedian(participant_median_movement_onset_time[participant_ids[i,:]], axis=0),
                np.nanmedian(participant_sd_movement_onset_time[participant_ids[i,:]], axis=0),
                np.nanmedian(participant_wins[participant_ids[i,:]],axis=0)/it.num_trials,
                np.nanmedian(participant_incorrects[participant_ids[i,:]],axis=0)/it.num_trials,
                np.nanmedian(participant_indecisions[participant_ids[i,:]],axis=0)/it.num_trials,
            ]   
        )      
    model_metric_keys = ['wtd_leave_time','wtd_leave_time_sd','prob_win','prob_incorrect','prob_indecision']
    model_fit_object = ModelFitting(model=fit_model)
    # start_time = time.time()
    res = model_fit_object.run_model_fit_procedure(
        free_params_init=free_params,
        targets=comparison_targets,
        drop_condition_from_loss=None,  # Drop 1200 50
        metric_keys=model_metric_keys,
        bnds=None,
        xtol=1e-6,
        ftol =1e-6,
        method="Powell",
        maxiter=5,
        maxfev = 300,
    )
    # end_time = time.time()
    # print(f"Time: {end_time - start_time}")
    specific_model_name = specific_name + model_name
    loss = model_fit_object.loss_store[-1]
    input_row_dict = create_input_row_dict(fit_model, loss, specific_model_name,list(free_params.keys()))
    input_parameters_for_df.append(input_row_dict)
    results_dict   = create_results_row_dict(fit_model,loss,specific_model_name,list(free_params.keys()))
    results_for_df.append(results_dict)
    if not WARM_START and STORE_BASE_MODEL:
        base_model_name = "base_" + model_name
        base_model_fit_object = ModelFitting(model=optimal_model_no_switch)
        base_model_loss = base_model_fit_object.get_loss(free_params.values(),model_metric_keys,comparison_targets,free_params.keys(),update=False,)
        
        base_input_row_dict = create_input_row_dict(optimal_model_no_switch, base_model_loss, base_model_name,list(free_params.keys()))
        base_input_parameters_for_df.append(base_input_row_dict)
        base_results_dict   = create_results_row_dict(optimal_model_no_switch,base_model_loss,base_model_name,list(free_params.keys()))
        base_results_for_df.append(base_results_dict)
        
df_inputs = pd.DataFrame(input_parameters_for_df)
df_results = pd.DataFrame(results_for_df)

if SAVE:
    save_date = datetime.now()
    if WARM_START:
        n = "warmstart"
    else:
        n = "bootstrapped"
        #* Save base model if we aren't warmstarting and if we want to
        if STORE_BASE_MODEL:
            with open(constants.MODELS_PATH / f"{n}_models" / f"{EXPERIMENT}_{specific_name}base_inputs_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
                dill.dump(df_inputs, f)
            with open(constants.MODELS_PATH / f"{n}_models" / f"{EXPERIMENT}_{specific_name}base_results_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
                dill.dump(df_results, f)
                
    #* Save either warmstart or bootstrapped
    with open(constants.MODELS_PATH / f"{n}_models" / f"{EXPERIMENT}_{specific_name}{n}_inputs_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
        dill.dump(df_inputs, f)
    with open(constants.MODELS_PATH / f"{n}_models" / f"{EXPERIMENT}_{specific_name}{n}_results_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
        dill.dump(df_results, f)
    
        
finish_time = time.time()
total_time = finish_time - initial_time
print(f"Model generation for {EXPERIMENT} completed")
print(f"Total Runtime for {iters} iterations: {total_time}")
