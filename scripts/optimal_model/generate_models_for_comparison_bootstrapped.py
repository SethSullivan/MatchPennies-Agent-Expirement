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
from Optimal_Stopping_Object import ModelConstructor, ModelFitting
from initializer import InitialThangs
import loss_functions as lf
import constants

'''
This script can either do the warmstarts (set WARM_START = True)

It can also bootstrap using the best warmstart initial conditions
'''

# * Functions
def create_input_row_dict(model, loss, model_name,):
    input_row_dict = {"Model": model_name, "Loss": loss}
    input_row_dict.update(vars(model.inputs))
    input_row_dict.pop("timesteps")

    return input_row_dict

def create_results_row_dict(model,loss,model_name):
    get_metric = model.results.get_metric
    model_data = [
        get_metric(model.player_behavior.wtd_leave_time,metric_type='true',decision_type='optimal'),
        get_metric(model.player_behavior.wtd_leave_time_sd,metric_type='true',decision_type='optimal'),
        get_metric(model.score_metrics.prob_indecision,metric_type='true',decision_type='optimal')*100,
        get_metric(model.score_metrics.prob_win,metric_type='true',decision_type='optimal')*100,
        get_metric(model.score_metrics.prob_incorrect,metric_type='true',decision_type='optimal')*100,
    ]
    results_row = {
        "Model":model_name,"Loss":loss,
        "median_movement_onset_time":model_data[0],
        "sd_movement_onset_time":model_data[1],
        "indecisions":model_data[2],
        "wins":model_data[3],
        "incorrects":model_data[4],
    }
    return results_row

#* GLOBAL PARAMETERS
# * Select experiment you'd like to run
EXPERIMENT = "Exp1"
LOAD_PATH = Path(f"D:\OneDrive - University of Delaware - o365\Subject_Data\MatchPennies_Agent_{EXPERIMENT}")
it = InitialThangs(EXPERIMENT)

#* DECIDE TO FIT PARAMETERS OR NOT
FIT_PARAMETERS = True
SAVE = True
M = 100
MODEL_TO_FIT = "suboptimal"
print(f"Fitting {MODEL_TO_FIT}")
WARM_START = True
input_keys = ["rt","rt_sd","mt","mt_sd","timing_sd",]




with open(constants.MODELS_PATH / 'model_input_dict.pkl','rb') as f:
    model_input_dict = dill.load(f)

true_parameters = [np.nanmedian(v) for k,v in model_input_dict.items() if "agent" not in k] 

with open(constants.MODEL_INPUT_PATH / 'bootstrap_parameter_distribution.pkl','rb') as f:
    parameter_distribution = dill.load(f)    
with open(constants.MODEL_INPUT_PATH / 'bootstrap_results.pkl','rb') as f:
    results = dill.load(f)    
with open(constants.MODEL_INPUT_PATH / 'participant_ids.pkl','rb') as f:
    participant_ids = dill.load(f)        
    
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
    
input_parameters_for_df = []
results_for_df = []
#* Load in warmstart if it exists, won't use this unless WARM_START = False
try:
    path = constants.MODELS_PATH / "warmstart_models"
    f_results = list(path.glob(f"{EXPERIMENT}_{MODEL_TO_FIT}_warmstart_results*"))[-1]
    f_inputs = list(path.glob(f"{EXPERIMENT}_{MODEL_TO_FIT}_warmstart_inputs*"))[-1]
    df_warmstart_results = pd.read_pickle(path / f_results)
    df_warmstart_inputs = pd.read_pickle(path / f_inputs)
except:
    pass

if WARM_START:
    print("FINDING INITIAL CONDITIONS")
    iters = 3
else:
    print("BOOTSTRAPPING MODEL FITS USING WARMSTART")
    iters = participant_ids.shape[1]
initial_time = time.time()
print("Starting Models...")
for i in tqdm(range(iters)):
    print(f"{i+1}/{iters}")
    #* 
    if WARM_START:
        player_inputs = dict(zip(input_keys,true_parameters))
        switch_delay_rand = np.random.uniform(0,150)
        switch_sd_rand = np.random.uniform(0,200)
        #* Optimal/Suboptimal handles the initial guess to free params mapping
        initial_guess = {
            "guess_switch_delay_true": switch_delay_rand + np.random.uniform(0,75),
            "guess_switch_delay_expected": switch_delay_rand,
            "guess_switch_sd_true": switch_sd_rand + np.random.uniform(0,75),
            "guess_switch_sd_expected": switch_sd_rand,
        }
    else: 
        #TODO Load in initial guess dict from warmstart
        player_inputs = dict(zip(input_keys,parameter_distribution[i,:]))


    # * Loop through all the changing parameters
    model_name = f"model{i}_{datetime.now():%Y_%m_%d_%H_%M_%S}"

    #* 3 Models 
    #* 1. Full optimal, no fitting, no switch delay or uncertainty
    #* 2. Full optimal, accounting for fit switch delay and uncertainty, and the expected and true have to be equal
    #* 3. Full optimal, not accounting for fit switch delay and uncertainty, and the expected and true are both fit simultaneously
    s = time.time()
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
    if MODEL_TO_FIT == "optimal":
        fit_model = deepcopy(optimal_model_no_switch)
        #! Not putting _true or _expected makes true == expected
        free_params = {
            "guess_switch_delay": initial_guess['guess_switch_delay_true'],
            "guess_switch_sd": initial_guess['guess_switch_sd_true'],
            }
        specific_name = 'optimal_'
        
    elif MODEL_TO_FIT == "suboptimal":
        fit_model = ModelConstructor(
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
        #* Fit the true and expected separately and see what the model does
        free_params = {
            "guess_switch_delay_true": initial_guess['guess_switch_delay_true'],
            "guess_switch_delay_expected": initial_guess['guess_switch_delay_expected'],
            "guess_switch_sd_true": initial_guess['guess_switch_sd_true'],
            "guess_switch_sd_expected": initial_guess['guess_switch_sd_expected'],
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
                    
    # Need to be in for loop because we're using specific participant_ids
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
        # comparison_targets = np.array(
        #     [
        #         np.nanmedian(participant_median_movement_onset_time, axis=0),
        #         np.nanmedian(participant_sd_movement_onset_time, axis=0),
        #     ]
        # )
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
    metric_keys = ['wtd_leave_time','wtd_leave_time_sd','prob_win','prob_incorrect','prob_indecision']
    model_fit = ModelFitting(model=fit_model)
    start_time = time.time()
    res = model_fit.run_model_fit_procedure(
        free_params_init=free_params,
        targets=comparison_targets,
        drop_condition_from_loss=None,  # Drop 1200 50
        limit_sd=False,
        metric_keys=metric_keys,
        bnds=None,
        tol=0.0001,
        method="Powell",
    )
    end_time = time.time()
    print(f"Time: {end_time - start_time}")
    specific_model_name = specific_name + model_name
    loss = model_fit.loss_store[-1]
    input_row_dict = create_input_row_dict(fit_model, loss, specific_model_name)
    input_parameters_for_df.append(input_row_dict)
    results_dict   = create_results_row_dict(fit_model,loss,specific_model_name)
    results_for_df.append(results_dict)
        
df_inputs = pd.DataFrame(input_parameters_for_df)
df_results = pd.DataFrame(results_for_df)

if SAVE:
    if WARM_START:
        save_date = datetime.now()
        # * Save the old model table to a new file
        with open(constants.MODELS_PATH / "warmstart_models" / f"{EXPERIMENT}_{specific_name}warmstart_inputs_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
            dill.dump(df_inputs, f)
        with open(constants.MODELS_PATH / "warmstart_models"/ f"{EXPERIMENT}_{specific_name}warmstart_results_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
            dill.dump(df_results, f)
finish_time = time.time()
total_time = finish_time - initial_time
print(f"Model generation for {EXPERIMENT} completed")
print(f"Total Runtime for {iters} number of iterations: {total_time}")
