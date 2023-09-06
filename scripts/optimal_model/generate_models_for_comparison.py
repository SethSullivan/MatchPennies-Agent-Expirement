import numpy as np
import dill
import matplotlib.pyplot as plt
import data_visualization as dv
from copy import deepcopy
import time
from pathlib import Path
import pandas as pd

import read_data_functions as rdf
import plot_functions as pf
from Optimal_Stopping_Object import ModelConstructor
from initializer import InitialThangs
import loss_functions as lf
%load_ext autoreload
%autoreload 2

#* Select experiment you'd like to run
EXPERIMENT = "Exp1"

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
    if experiment == "Exp1":
        rt    = np.nanmedian(np.nanmedian(group.movement_metrics.reaction_times, axis=1)) - 25
        rt_sd = np.nanmedian(np.nanstd(group.movement_metrics.reaction_times, axis=1))

    elif experiment == "Exp2":
        rt    = np.nanmedian(np.nanmedian(group.movement_metrics.exp2_react_guess_reaction_time_split('react','only'), axis=1)) - 25
        rt_sd = np.nanmedian(np.nanstd(group.movement_metrics.reaction_times, axis=2))

    mt                   = np.min(np.nanmedian(np.nanmedian(group.movement_metrics.movement_times('task'), axis=2), axis=0)) # Get movement time for the condition where they tried the hardest
    mt_sd                = np.nanmedian(np.nanstd(group.movement_metrics.movement_times('task'), axis=1))
    time_sd              = np.array([np.nanmedian(np.nanstd(group.movement_metrics.coincidence_reach_time, axis=1))] * it.num_blocks)
    perc_wins_both_reach = np.nanmean(group.score_metrics.wins_when_both_reach(perc=True), axis=0)
    guess_sd             = np.nanmedian(np.nanstd(group.react_guess_movement_metrics.movement_onset_times('guess'), axis=2), axis=0)
    agent_sds            = np.nanmean(np.nanstd(group.raw_data.agent_task_leave_time, axis=2), axis=0)[:,np.newaxis]
    agent_means          = np.nanmean(np.nanmean(group.raw_data.agent_task_leave_time, axis=2), axis=0)[:,np.newaxis]
    guess_leave_time_sd  = np.nanmedian(np.nanstd(group.react_guess_movement_metrics.movement_onset_times('guess'),axis=2),axis=0)


#* Set the parameters that change with each model
if True: 
    AGENT_SD_CHANGE  = 150
    RT_SD_CHANGE     = rt_sd/2
    MT_SD_CHANGE     = mt_sd/2
    TIMING_SD_CHANGE = time_sd/2
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
    score_rewards_list = [(1.0, 0.0, 0.0), (1.0, -0.2, 0.0)]

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
metric_keys = ['wtd_leave_time','prob_win','prob_incorrect','prob_indecision']

#* Loop through all the changing parameters
for i,(agent_sd_change, rt_sd_change, mt_sd_change, timing_sd_change) in enumerate(change_sd_list):
    for j, (guess_switch_delay_true, guess_switch_delay_expected) in enumerate(zip(guess_switch_delay_true_list,guess_switch_delay_expected_list)):
        for k, (guess_switch_sd_true, guess_switch_sd_expected) in enumerate(zip(guess_switch_sd_true_list,guess_switch_sd_expected_list)):
            for m, (win_reward,incorrect_cost,indecision_cost) in enumerate(score_rewards_list):
                model  = ModelConstructor(
                    experiment=EXPERIMENT,
                    num_blocks=it.num_blocks,
                    num_timesteps=1800,
                    agent_means= np.array([agent_means,agent_means]),
                    agent_sds= np.array([agent_sds,agent_sds + agent_sd_change]), #!
                    reaction_time=np.array([rt, rt]),
                    movement_time=np.array([mt, mt]),
                    reaction_sd  =np.array([rt_sd, rt_sd - rt_sd_change]), #! Reducing these, aka the particiapnt thinks they are more certain than they are
                    movement_sd  =np.array([mt_sd, mt_sd - mt_sd_change]),
                    timing_sd    =np.array([time_sd, time_sd - timing_sd_change]),
                    perc_wins_when_both_reach=perc_wins_both_reach,
                    guess_switch_delay=np.array([[guess_switch_delay_true, guess_switch_delay_expected]]).T, # Designed like this for broadcasting reasons
                    guess_sd     = np.array([guess_switch_sd_true,guess_switch_sd_expected]), # This includes electromechanical delay sd and timing sd bc it's straight from data
                    electromechanical_delay=np.array([[50, 50]]).T,
                    switch_cost_exists=True,
                    expected=True,
                    win_reward=win_reward,
                    incorrect_cost=incorrect_cost,
                    indecision_cost=indecision_cost,
                )
                inputs_row = vars(model.inputs)
                model_metrics = [model.player_behavior.wtd_leave_time, model.score_metrics.prob_win,
                                 model.score_metrics.prob_incorrect,model.score_metrics.prob_indecision ]
                predictions = [model.results.get_metric(metric,decision_type='optimal',metric_type='true') for metric in model_metrics]
                loss = lf.ape_loss(predictions, targets, drop_condition_num=None) 
                
                
    # known_switch_delay_no_altered_reward_dict.update({model_dict_keys_sd[i]:model})

