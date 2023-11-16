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
import constants

'''
This script knockouts the "known" parameter for 
1. Reaction Time Mean
2. Reaction Time SD
3. Movement Time Mean
4. Movement Time SD
5. Timing SD
6. Electromechanical Mean
7. Electromechanical SD
'''

# * Functions
def get_loss(model, targets, drop_condition_num=None):
    model_metrics = [
        model.player_behavior.wtd_leave_time,
        model.player_behavior.wtd_leave_time_sd,
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


# * Select experiment you'd like to run
EXPERIMENT = "Exp1"
BOOTSTRAP = True
SAVE = True

print(f'Starting up {EXPERIMENT}')
# * GET THE MODEL TRACKER TABLE
MODELS_PATH = Path(f"D:\\OneDrive - University of Delaware - o365\\Desktop\\MatchPennies-Agent-Expirement\\results\\models")

# * Initial Thangs
# Get path and save path
LOAD_PATH = Path(f"D:\OneDrive - University of Delaware - o365\Subject_Data\MatchPennies_Agent_{EXPERIMENT}")
it = InitialThangs(EXPERIMENT)

#* Model input, true parameters
with open(constants.MODELS_PATH / 'model_input_dict.pkl','rb') as f:
    model_input_dict = dill.load(f)
true_parameters = {k:np.nanmedian(v) for k,v in model_input_dict.items() if "agent" not in k} 

#* Bootstrap parameters
with open(constants.MODEL_INPUT_PATH / 'bootstrap_parameter_distribution.pkl','rb') as f:
    parameter_distribution = dill.load(f)    
with open(constants.MODEL_INPUT_PATH / 'bootstrap_results.pkl','rb') as f:
    results = dill.load(f)    
with open(constants.MODEL_INPUT_PATH / 'participant_ids.pkl','rb') as f:
    participant_ids = dill.load(f)

#* Particiapnt results, used for loss function 
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
    
# * Set the parameters that change with each model
# Create keys of what's being changed
model_dict_keys_sd = ["agent_sd", "rt_sd", "mt_sd", "timing_sd"]
if BOOTSTRAP:
    iters = participant_ids.shape[0]
else:
    params_dict = {
        "rt_true":true_parameters['rt'],
        "rt_expected":true_parameters['rt'],
        "rt_sd_true":true_parameters['rt_sd'],
        "rt_sd_expected":true_parameters['rt_sd'],
        "mt_true":true_parameters['mt'],
        "mt_expected":true_parameters['mt'],
        "mt_sd_true":true_parameters['mt_sd'],
        "mt_sd_expected":true_parameters['mt_sd'],
        "timing_sd_true":true_parameters['timing_sd'],
        "timing_sd_expected":true_parameters['timing_sd'],
        "electromechanical_true":50,
        "electromechanical_expected":50,
        "electromechanical_sd_true":50,
        "electromechanical_sd_expected":50,
    }
    param_keys = params_dict.keys()

# * Get targets for model comparisons
comparison_targets = np.array(
    [
        np.nanmedian(participant_median_movement_onset_time, axis=0),
        np.nanmedian(participant_sd_movement_onset_time, axis=0),
        np.nanmedian(participant_wins,axis=0)/it.num_trials,
        np.nanmedian(participant_incorrects,axis=0)/it.num_trials,
        np.nanmedian(participant_indecisions,axis=0)/it.num_trials,
    ]   
)
# * Loop through all the changing parameters
c = 0
input_parameters = []
descriptive_parameters = []  # Used for saying what changed, as opposed to the actual parameter values
print("Starting Models...")
unaccounted_for_parameter = ["rt","rt_sd","mt","mt_sd",
                             "timing_sd","electromechanical","electromechanical_sd"]
input_keys = ["rt","rt_sd","mt","mt_sd","timing_sd",]

for parameter in unaccounted_for_parameter:
    for i in tqdm(range(iters)):
        if BOOTSTRAP:
            parameters = dict(zip(input_keys,parameter_distribution[i,:]))
            params_dict = params_dict = {
                            "rt_true":parameters['rt'],
                            "rt_expected":parameters['rt'],
                            "rt_sd_true":parameters['rt_sd'],
                            "rt_sd_expected":parameters['rt_sd'],
                            "mt_true":parameters['mt'],
                            "mt_expected":parameters['mt'],
                            "mt_sd_true":parameters['mt_sd'],
                            "mt_sd_expected":parameters['mt_sd'],
                            "timing_sd_true":parameters['timing_sd'],
                            "timing_sd_expected":parameters['timing_sd'],
                            "electromechanical_true":50,
                            "electromechanical_expected":50,
                            "electromechanical_sd_true":10,
                            "electromechanical_sd_expected":10,
                        }
            
        #* Set expected parameter to 1, meaning it's not accounted for
        params_dict[parameter+"_expected"] = 1
        model_name = f"model{c}_{datetime.now():%Y_%m_%d_%H_%M_%S}"

        descriptive_parameter_row = {
            "Model": model_name,
            "Loss": 0,
            "Known_RT": params_dict["rt_expected"]==params_dict["rt_true"],
            "Known_RT_SD": params_dict["rt_sd_expected"]==params_dict["rt_sd_true"],
            "Known_MT": params_dict["mt_expected"]==params_dict["mt_true"],
            "Known_MT_SD": params_dict["mt_sd_expected"]==params_dict["mt_sd_true"],
            "Known_Timing_SD":params_dict["timing_sd_expected"]==params_dict["timing_sd_true"],
            "Known_Electromechanical": params_dict["electromechanical_expected"]==params_dict["electromechanical_true"],
            "Known_Electromechanical_SD": params_dict["electromechanical_sd_expected"]==params_dict["electromechanical_sd_true"],
        }
        descriptive_parameters.append(descriptive_parameter_row)
        
        model = ModelConstructor(
            experiment=EXPERIMENT,
            num_blocks=it.num_blocks,
            num_timesteps=1800,
            agent_means=np.array([model_input_dict["agent_means"], model_input_dict["agent_means"]])[:, :, np.newaxis],
            agent_sds=np.array([model_input_dict["agent_sds"], model_input_dict["agent_sds"]])[:, :, np.newaxis],  #!
            reaction_time=np.array([params_dict["rt_true"], params_dict["rt_expected"]])[:, np.newaxis, np.newaxis],
            movement_time=np.array([params_dict["mt_true"], params_dict["mt_expected"]])[:, np.newaxis, np.newaxis],
            reaction_sd=np.array([params_dict["rt_sd_true"], params_dict["rt_sd_expected"]])[
                :, np.newaxis, np.newaxis
            ],  #! Reducing these, aka the particiapnt thinks they are more certain than they are
            movement_sd=np.array([params_dict["mt_sd_true"], params_dict["mt_sd_expected"]])[:, np.newaxis, np.newaxis],
            timing_sd=np.array([[params_dict["timing_sd_true"]]*it.num_blocks, [params_dict["timing_sd_expected"]]*it.num_blocks])[:, :, np.newaxis],
            guess_switch_delay=np.array([0,0])[:, np.newaxis, np.newaxis],  # Designed like this for broadcasting reasons
            guess_switch_sd=np.array([0,0])[:, np.newaxis, np.newaxis],  # This includes electromechanical delay sd and timing sd bc it's straight from data
            electromechanical_delay=np.array([params_dict["electromechanical_true"], params_dict["electromechanical_expected"]])[:, np.newaxis, np.newaxis],
            electromechanical_sd=np.array([params_dict["electromechanical_sd_true"], params_dict["electromechanical_sd_expected"]])[:, np.newaxis, np.newaxis],
            switch_cost_exists=True,
            expected=True,  #! Should always be True... if the parameter is ground truth, then the two values of the parameter array should be the same
            win_reward=1.0,
            incorrect_cost=0.0,  #! These are applied onto the base reward matrix in Optimal Model object
            indecision_cost=0.0,
            round_num = 20,
        )
        loss = get_loss(
            model,
            comparison_targets,
        )
        descriptive_parameter_row['Loss'] = loss
        input_row_dict = create_input_row_dict(model, loss, model_name)
        input_parameters.append(input_row_dict)
        c += 1
        
        #* Set expected parameter back to true
        params_dict[parameter+"_expected"] = params_dict[parameter+"_true"]

df_inputs = pd.DataFrame(input_parameters)
df_descriptions = pd.DataFrame(descriptive_parameters)
if SAVE:
    save_date = datetime.now()
    # * Save the old model table to a new file
    with open(MODELS_PATH / "knockout_models" / f"{EXPERIMENT}_bootstrapped_model_parameters_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
        dill.dump(df_inputs, f)

    with open(MODELS_PATH / "knockout_models" / f"{EXPERIMENT}_bootstrapped_model_descriptions_{save_date:%Y_%m_%d_%H_%M_%S}.pkl", "wb") as f:
        dill.dump(df_descriptions, f)

print(f"Model generation for {EXPERIMENT} completed")
