import constants
import numpy as np
import pandas as pd
import dill
import model_helper_functions as mhf

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
        get_metric(model.player_behavior.wtd_leave_time,metric_type='true',decision_type='optimal'),
        get_metric(model.player_behavior.wtd_leave_time_sd,metric_type='true',decision_type='optimal'),
        get_metric(model.score_metrics.prob_indecision,metric_type='true',decision_type='optimal')*100,
        get_metric(model.score_metrics.prob_win,metric_type='true',decision_type='optimal')*100,
        get_metric(model.score_metrics.prob_incorrect,metric_type='true',decision_type='optimal')*100,
    ]
    results_row = {
        "Model":model_name,"Loss":loss,"fit_parameters":free_param_keys,
        "decision_times":model_data[0],
        "target_reach_times":model_data[1],
        "target_reach_times_sd":model_data[2],
        "median_movement_onset_time":model_data[3],
        "sd_movement_onset_time":model_data[4],
        "indecisions":model_data[5],
        "wins":model_data[6],
        "incorrects":model_data[7],
    }
    return results_row

EXPERIMENT = "Exp1"
path = constants.MODELS_PATH / f"bootstrapped_models"
model_names = ["Base","Optimal","Suboptimal_All"]
model_results = []
model_inputs = []
for model in model_names:
    # Load models
    model_results = pd.DataFrame()
    inputs_path = list(path.glob(f"{EXPERIMENT}_{model.lower()}_bootstrapped_inputs*"))[-1]
    model_df = pd.read_pickle(path / inputs_path)
    models_dict = mhf.run_models_from_df(model_df.iloc[:10])

    
    
    
# #* Run through each model using model_inputs
# model_object_dicts = []
# for i,model_df in enumerate(model_inputs):
#     for model_name, model in models_dict.keys():
        
#     break

        