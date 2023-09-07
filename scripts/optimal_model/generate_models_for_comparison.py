import numpy as np
import dill
from copy import deepcopy
from pathlib import Path
import pandas as pd
from datetime import datetime

import read_data_functions as rdf
from Optimal_Stopping_Object import ModelConstructor
from initializer import InitialThangs
import loss_functions as lf

#* Select experiment you'd like to run
EXPERIMENT = "Exp1"

#* GET THE MODEL TRACKER TABLE
MODELS_PATH = Path(r'D:\OneDrive - University of Delaware - o365\Desktop\MatchPennies-Agent-Expirement\results\exp1\models')

#* Initial Thangs
# Get path and save path 
LOAD_PATH = Path(f"D:\OneDrive - University of Delaware - o365\Subject_Data\MatchPennies_Agent_{EXPERIMENT}")
SAVE_PATH = f"D:\\OneDrive - University of Delaware - o365\\Subject_Data\\MatchPennies_Agent_{EXPERIMENT}\\Figures\\"
it = InitialThangs(EXPERIMENT)

#* Get group data
if 'group' not in locals():
    group = rdf.generate_subject_object_v3(EXPERIMENT,'All Trials')
    
#* Set inputs for models
if True:
    if EXPERIMENT == "Exp1":
        rt    = np.nanmedian(np.nanmedian(group.movement_metrics.reaction_times, axis=1)) - 25
        rt_sd = np.nanmedian(np.nanstd(group.movement_metrics.reaction_times, axis=1))

    elif EXPERIMENT == "Exp2":
        rt    = np.nanmedian(np.nanmedian(group.movement_metrics.exp2_react_guess_reaction_time_split('react','only'), axis=1)) - 25
        rt_sd = np.nanmedian(np.nanstd(group.movement_metrics.reaction_times, axis=2))

    mt                   = np.min(np.nanmedian(np.nanmedian(group.movement_metrics.movement_times('task'), axis=2), axis=0)) # Get movement time for the condition where they tried the hardest
    mt_sd                = np.nanmedian(np.nanstd(group.movement_metrics.movement_times('task'), axis=1))
    time_sd              = np.array([np.nanmedian(np.nanstd(group.movement_metrics.coincidence_reach_time, axis=1))] * it.num_blocks) 
    perc_wins_both_reach = np.nanmean(group.score_metrics.wins_when_both_reach(perc=True), axis=0)
    guess_sd             = np.nanmedian(np.nanstd(group.react_guess_movement_metrics.movement_onset_times('guess'), axis=2), axis=0)
    agent_sds            = np.nanmean(np.nanstd(group.raw_data.agent_task_leave_time, axis=2), axis=0)
    agent_means          = np.nanmean(np.nanmean(group.raw_data.agent_task_leave_time, axis=2), axis=0)
    guess_leave_time_sd  = np.nanmedian(np.nanstd(group.react_guess_movement_metrics.movement_onset_times('guess'),axis=2),axis=0)


#* Set the parameters that change with each model
if True:
    AGENT_SD_CHANGE  = 150
    RT_SD_CHANGE     = rt_sd/2
    MT_SD_CHANGE     = mt_sd/2
    TIMING_SD_CHANGE = time_sd[0]/2 # Array of 6, just want one
    GUESS_SWITCH_DELAY = 65
    GUESS_SWITCH_SD    = 30
    # Create keys of what's being changed
    model_dict_keys_sd = ['agent_sd', 'rt_sd', 'mt_sd', 'timing_sd']

    # (*LOOP 1*) List for altering each uncertainty
    change_sd_list = [(AGENT_SD_CHANGE,0,0,0), (0,RT_SD_CHANGE,0,0), (0,0,MT_SD_CHANGE,0), (0,0,0,TIMING_SD_CHANGE)]
    # (*LOOP 2*) Create guess switch delay true and expected, where every other is equal
    guess_switch_delay_true_list = [GUESS_SWITCH_DELAY]*2
    guess_switch_delay_expected_list = [GUESS_SWITCH_DELAY,0]

    # (*LOOP 3*) Create guess switch sd true and expected, where every other is equal
    guess_switch_sd_true_list = [GUESS_SWITCH_SD]*2
    guess_switch_sd_expected_list = [GUESS_SWITCH_SD,0]

    # (*LOOP 4*)
    if EXPERIMENT == 'Exp1':  
        BASE_WIN_REWARD = 1.0
        BASE_INCORRECT_COST = 0.0
        BASE_INDECISION_COST = 0.0
        score_rewards_list = np.array([[1.0, 0.0, 0.0], [1.0, -0.2, 0.0]])
        reward_matrix_list = [np.array([[1, 0, 0], [1, -1, 0], [1, 0, -1], [1, -1, -1]])] #! Not used in exp1, so just one element in loop

    if EXPERIMENT == 'Exp2':  
        score_rewards_list = np.array([[1.0, 0.0, 0.0], [1.0, -0.2, 0.0]])

        reward_matrix_list = [np.array([[1, 0, 0], [1, -1, 0], [1, 0, -1], [1, -1, -1]]),
                              np.array([[1, -0.2, 0], [1, -1.2, 0], [1, -0.2, -1], [1, -1.2, -1]])]

    ## Set numbers to change means by
    # agent_mean_change = 50
    # RT_SD_CHANGE     = 20
    # MT_SD_CHANGE     = 20
    # TIMING_SD_CHANGE = time_sd/2
    # change_sd_list = [(AGENT_SD_CHANGE,0,0,0), (0,RT_SD_CHANGE,0,0), (0,0,MT_SD_CHANGE,0), (0,0,0,TIMING_SD_CHANGE)]
    # model_dict_keys_sd = ['agent_sd', 'rt_sd', 'mt_sd', 'timing_sd']
    # assert guess_switch_delay_expected_list[::2] == guess_switch_delay_true_list[::2]
    
#* Get targets for model comparisons
targets = np.array(
    [np.nanmedian(np.nanmedian(group.movement_metrics.movement_onset_times('task'), axis=2),axis=0),
    np.nanmedian(group.score_metrics.score_metric('wins'),axis=0)/100,
    np.nanmedian(group.score_metrics.score_metric('incorrects'),axis=0)/100,
    np.nanmedian(group.score_metrics.score_metric('indecisions'),axis=0)/100,
    ]
)
# metric_keys = ['wtd_leave_time','prob_win','prob_incorrect','prob_indecision']


#* Functions
def get_loss(model, targets, drop_condition_num=None):
    model_metrics = [model.player_behavior.wtd_leave_time, model.score_metrics.prob_win,
                    model.score_metrics.prob_incorrect,model.score_metrics.prob_indecision]
    predictions = [model.results.get_metric(metric,decision_type='optimal',metric_type='true') for metric in model_metrics]
    loss = lf.ape_loss(predictions, targets, drop_condition_num=drop_condition_num) 
    
    return loss

def create_input_row_dict(model, loss, model_name,):
    input_row_dict = ({'Model Name':model_name,
                       'Loss':loss})
    input_row_dict.update(vars(model.inputs))
    input_row_dict.pop('timesteps')
    
    return input_row_dict

def map_reward_change(score:float, comparison_num:float) -> str:
    if score>comparison_num:
        score_change = 'Greater'
    elif score<comparison_num:
        score_change = 'Less'
    else:
        score_change = 'Normal'
    return score_change

#* Loop through all the changing parameters
c=0
list_of_input_parameters = []
list_of_descriptive_parameters    = [] # Used for saying what changed, as opposed to the actual parameter values
print('Starting Models...')
for i,(agent_sd_change, rt_sd_change, mt_sd_change, timing_sd_change) in enumerate(change_sd_list):
    for j, (guess_switch_delay_true, guess_switch_delay_expected) in enumerate(zip(guess_switch_delay_true_list,guess_switch_delay_expected_list)):
        for k, (guess_switch_sd_true, guess_switch_sd_expected) in enumerate(zip(guess_switch_sd_true_list,guess_switch_sd_expected_list)):
            for m, (win_reward,incorrect_cost,indecision_cost) in enumerate(score_rewards_list):
                for n, (reward_matrix) in enumerate(reward_matrix_list):
                    model  = ModelConstructor(
                        experiment=EXPERIMENT,
                        num_blocks=it.num_blocks,
                        num_timesteps=1800,
                        agent_means=np.array([agent_means,agent_means])[:,:,np.newaxis],
                        agent_sds=np.array([agent_sds,agent_sds + agent_sd_change])[:,:,np.newaxis], #!
                        reaction_time=np.array([rt, rt])[:,np.newaxis,np.newaxis],
                        movement_time=np.array([mt, mt])[:,np.newaxis,np.newaxis],
                        reaction_sd=np.array([rt_sd, rt_sd - rt_sd_change])[:,np.newaxis,np.newaxis], #! Reducing these, aka the particiapnt thinks they are more certain than they are
                        movement_sd=np.array([mt_sd, mt_sd - mt_sd_change])[:,np.newaxis,np.newaxis],
                        timing_sd=np.array([time_sd, time_sd - timing_sd_change])[:,:,np.newaxis],
                        guess_switch_delay=np.array([guess_switch_delay_true, guess_switch_delay_expected])[:,np.newaxis,np.newaxis], # Designed like this for broadcasting reasons
                        guess_switch_sd=np.array([guess_switch_sd_expected,guess_switch_sd_expected])[:,np.newaxis,np.newaxis], # This includes electromechanical delay sd and timing sd bc it's straight from data
                        electromechanical_delay=np.array([50, 50])[:,np.newaxis,np.newaxis],
                        switch_cost_exists=True,
                        expected=True, #! Should always be True... if the parameter is ground truth, then the two values should be the same
                        win_reward=win_reward,
                        incorrect_cost=incorrect_cost, #! These are applied onto the base reward matrix in Optimal Model object
                        indecision_cost=indecision_cost,
                    )
                    model_name = f'model{c}_{datetime.now():%Y_%m_%d_%H_%M_%S}'

                    loss = get_loss(model, targets,)
                    input_row_dict = create_input_row_dict(model, loss, model_name)
                    list_of_input_parameters.append(input_row_dict)  
                                       
                    descriptive_parameter_row = {
                        'Model':model_name,
                        'Loss':loss,
                        'Known Switch Delay':guess_switch_delay_expected == guess_switch_delay_true,
                        'Known Switch SD':guess_switch_sd_expected == guess_switch_sd_true,
                        'Known Agent SD':agent_sd_change==0,
                        'Known RT SD':rt_sd_change==0,
                        'Known MT SD':mt_sd_change==0,
                        'Known Timing SD':timing_sd_change==0,
                        'Win Reward':map_reward_change(win_reward,comparison_num=1.0),
                        'Incorrect Cost':map_reward_change(incorrect_cost,comparison_num=0.0),
                        'Indecision Cost':map_reward_change(indecision_cost,comparison_num=0.0),
                    }
                    list_of_descriptive_parameters.append(descriptive_parameter_row)
                    c+=1

df_inputs = pd.DataFrame(list_of_input_parameters)
df_descriptions = pd.DataFrame(list_of_descriptive_parameters)

save_date = datetime.now()
#* Save the old model table to a new file
with open(MODELS_PATH / f'{EXPERIMENT}_model_parameters_{save_date:%Y_%m_%d_%H_%M_%S}.pkl','wb') as f:
    dill.dump(df_inputs, f)

with open(MODELS_PATH / f'{EXPERIMENT}_model_descriptions_{save_date:%Y_%m_%d_%H_%M_%S}.pkl','wb') as f:
    dill.dump(df_descriptions, f)

print(f'Model generation for {EXPERIMENT} completed')
