
from pathlib import Path
import numpy as np
from Optimal_Stopping_Object import ModelConstructor
from initializer import InitialThangs
import read_data_functions as rdf

#* SELECT EXPERIMENT
experiment = "Exp1"

# Get path and save path 
LOAD_PATH = Path(f"D:\\OneDrive - University of Delaware - o365\\Subject_Data\\MatchPennies_Agent_{experiment}")
SAVE_PATH = f"D:\\OneDrive - University of Delaware - o365\\Subject_Data\\MatchPennies_Agent_{experiment}\\Figures\\"
it = InitialThangs(experiment)

group = rdf.generate_subject_object_v3(experiment,)

#* Inputs for model 
if experiment == "Exp1":
    rt = np.nanmedian(np.nanmedian(group.movement_metrics.reaction_times, axis=1)) - 20
    rt_sd = np.nanmedian(np.nanstd(group.movement_metrics.reaction_times, axis=1))

elif experiment == "Exp2":
    rt = np.nanmedian(np.nanmedian(group.movement_metrics.exp2_react_gamble_reaction_time_split('react','only'), axis=1)) - 30
    rt_sd = np.nanmedian(np.nanstd(group.movement_metrics.reaction_times, axis=2))

mt                   = np.min(np.nanmedian(np.nanmedian(group.movement_metrics.movement_times('task'), axis=2), axis=0)) # Get movement time for the condition where they tried the hardest
mt_sd                = np.nanmedian(np.nanstd(group.movement_metrics.movement_times('task'), axis=1))
time_sd              = np.array([np.nanmedian(np.nanstd(group.movement_metrics.coincidence_reach_time, axis=1))] * it.num_blocks)
perc_wins_both_reach = np.nanmean(group.score_metrics.wins_when_both_reach(perc=True), axis=0)
guess_sd             = np.nanmedian(np.nanstd(group.react_guess_movement_metrics.movement_onset_times('guess'), axis=2), axis=0)
agent_sds            = np.nanmean(np.nanstd(group.raw_data.agent_task_leave_time, axis=2), axis=0)[:,np.newaxis]
agent_means          = np.nanmean(np.nanmean(group.raw_data.agent_task_leave_time, axis=2), axis=0)[:,np.newaxis]

#* Set win, incorrect, indecision reward if messing around with beta parameters
win_reward = 1.0
indecision_cost = 0.0
incorrect_cost = 0.0

def test_true_and_expected_models_return_same_optimal_decision_time():
    '''
    ensuring that model with knowledge of delay and without knowledge of delay 
    have the same optimal decision time when the expected and true delays are 
    the same between them
    '''
    model_true = ModelConstructor(
        experiment=experiment,
        num_blocks=it.num_blocks,
        num_timesteps=5,
        BETA_ON=False,
        agent_means=agent_means,
        agent_sds= agent_sds,
        reaction_time=np.array([rt, rt]),
        movement_time=np.array([mt, mt]),
        reaction_sd  =np.array([rt_sd, rt_sd]),
        movement_sd  =np.array([mt_sd, mt_sd]), 
        timing_sd    =np.array([time_sd, time_sd]),
        perc_wins_when_both_reach=perc_wins_both_reach,
        gamble_switch_delay=np.array([[75, 0]]).T, # Designed like this for broadcasting reasons
        gamble_switch_sd=np.array([[25, 0]]).T,
        electromechanical_delay=np.array([[50, 50]]).T,
        electromechanical_sd=np.array([[10, 10]]).T,
        switch_cost_exists=True,
        expected=False,
        win_reward=win_reward,
        incorrect_cost=incorrect_cost,
        indecision_cost=indecision_cost,
    )
    model_expected = ModelConstructor(
        experiment=experiment,
        num_blocks=it.num_blocks,
        num_timesteps=5,
        BETA_ON=False,
        agent_means=agent_means,
        agent_sds= agent_sds,
        reaction_time=np.array([rt, rt]),
        movement_time=np.array([mt, mt]),
        reaction_sd  =np.array([rt_sd, rt_sd]),
        movement_sd  =np.array([mt_sd, mt_sd]), 
        timing_sd    =np.array([time_sd, time_sd]),
        perc_wins_when_both_reach=perc_wins_both_reach,
        gamble_switch_delay=np.array([[75, 0]]).T, # Designed like this for broadcasting reasons
        gamble_switch_sd=np.array([[25, 0]]).T,
        electromechanical_delay=np.array([[50, 50]]).T,
        electromechanical_sd=np.array([[10, 10]]).T,
        switch_cost_exists=True,
        expected=True, # ! THIS SHOULD BE THE ONLY THING DIFFERENT
        win_reward=win_reward,
        incorrect_cost=incorrect_cost,
        indecision_cost=indecision_cost,
    )
    
    assert np.all(model_expected.results.optimal_decision_time == model_true.results.optimal_decision_time)

# def test_get_moments_equations():
#     # Fake agent distribution 
#     values_agent = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
#     probs_agent  = np.array([0.1, 0.2, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
#     mean_agent = np.sum(values_agent*probs_agent)
#     std_agent = np.sqrt((values_agent-mean_agent)**2 * probs_agent) # Timing std of the fuck it time for the agent

#     # Fake player distribution
#     values_player = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
#     probs_player = np.array([0.0, 0.0, 0.1, 0.2, 0.4, 0.2, 0.1, 0.0, 0.0])
#     mean_player = np.sum(values_player*probs_player) # Timing mean of the fuck it time for the player
#     std_player = np.sqrt((values_player-mean_player)**2 * probs_player) # Timing std of the fuck it time for the player
    
#     timesteps = np.array((values_player,values_player))
#     time_means = deepcopy(timesteps[0])
    

if __name__ == '__main__':
    pass