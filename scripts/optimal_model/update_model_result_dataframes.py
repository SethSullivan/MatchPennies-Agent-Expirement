import constants
import numpy as np
import pandas as pd
import dill
import model_helper_functions as mhf
from tqdm import tqdm
from Optimal_Stopping_Object import ModelConstructor

'''
This script is for adding on any metrics I need from the fit models

1. Runs through each model using the model_inputs dataframe
2. Calculates Metrics
3. Stores Metrics

'''
def create_results_row_dict(model,loss,model_name,free_param_keys):
    get_metric = model.results.get_metric
    model_data = [
        model.results.optimal_decision_time,
        get_metric(model.player_behavior.wtd_reach_time,metric_type='true',decision_type='optimal'),
        get_metric(model.player_behavior.wtd_reach_time_sd,metric_type='true',decision_type='optimal'),
        get_metric(model.player_behavior.guess_reach_time,metric_type='true',decision_type='optimal'),
        model.player_behavior.guess_reach_time_sd.squeeze(),
        get_metric(model.player_behavior.reaction_reach_time,metric_type='true',decision_type='optimal'),
        get_metric(model.player_behavior.reaction_reach_time_sd,metric_type='true',decision_type='optimal'),
        get_metric(model.player_behavior.prob_selecting_guess,metric_type='true',decision_type='optimal'),
        get_metric(model.player_behavior.prob_selecting_reaction,metric_type='true',decision_type='optimal'),
        get_metric(model.player_behavior.wtd_leave_time,metric_type='true',decision_type='optimal'),
        get_metric(model.player_behavior.wtd_leave_time_sd,metric_type='true',decision_type='optimal'),
        get_metric(model.score_metrics.prob_indecision,metric_type='true',decision_type='optimal')*100,
        get_metric(model.score_metrics.prob_win,metric_type='true',decision_type='optimal')*100,
        get_metric(model.score_metrics.prob_incorrect,metric_type='true',decision_type='optimal')*100,
    ]
    results_keys = [
        "decision_times",
        "target_reach_times",
        "target_reach_times_sd",
        "target_reach_times_guess",
        "target_reach_times_guess_sd",
        "target_reach_times_react",
        "target_reach_times_react_sd",
        "prob_selecting_guess",
        "prob_selecting_reaction",
        "median_movement_onset_time",
        "sd_movement_onset_time",
        "indecisions",
        "wins",
        "incorrects",
        ]
    results_row = {"Model":model_name,"Loss":loss,"fit_parameters":free_param_keys} | dict(zip(results_keys, model_data))
    return results_row

def rerun_models_save_output(model_df, old_model_results):
    new_rows = []
    for index,row in tqdm(model_df.iterrows()):
        model  = ModelConstructor(
            experiment=row.experiment,
            num_blocks=row.num_blocks,
            num_timesteps=row.num_timesteps,
            agent_means=row.agent_means,
            agent_sds=row.agent_sds,
            reaction_time=row.reaction_time,
            movement_time=row.movement_time,
            reaction_sd=row.reaction_sd, 
            movement_sd=row.movement_sd,
            timing_sd=row.timing_sd,
            guess_switch_delay=row.guess_switch_delay, 
            guess_switch_sd=row.guess_switch_sd, 
            electromechanical_delay=row.electromechanical_delay,
            expected=row.expected,
            win_reward=row.win_reward,
            incorrect_cost=row.incorrect_cost,
            indecision_cost=row.indecision_cost,
            round_num = 20,
            use_agent_behavior_lookup=False,
        )
        model_name = row.Model
        old_results_row = old_model_results.query("Model == @model_name")
        new_rows.append(create_results_row_dict(model, old_results_row.iloc[0]['Loss'], old_results_row.iloc[0]["Model"], old_results_row.iloc[0]["fit_parameters"]))
    return new_rows

EXPERIMENT = "Exp1"
path = constants.MODELS_PATH / f"bootstrapped_models"
model_names = ["Base","Optimal","Suboptimal_Partial"]
model_results = []
model_inputs = []
for model in model_names:
    # Load models
    inputs_path = list(path.glob(f"{EXPERIMENT}_{model.lower()}_bootstrapped_inputs*"))[-1]
    print(inputs_path)
    old_model_inputs = pd.read_pickle(path / inputs_path)
    results_path = list(path.glob(f"{EXPERIMENT}_{model.lower()}_bootstrapped_results*"))[-1]
    old_model_results = pd.read_pickle(path / results_path)
    new_rows = []
    new_rows = rerun_models_save_output(old_model_inputs, old_model_results)
    new_model_results = pd.DataFrame(new_rows)
    with open(results_path.as_posix()[:-4] + "_new.pkl", "wb") as f:
        dill.dump(new_model_results, f)
    print("model done")

        