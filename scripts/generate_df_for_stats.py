import numpy as np
import pandas as pd
import dill
from pathlib import Path
from scipy.stats import iqr
from initializer import InitialThangs
import read_data_functions as rdf



def generate_exp1_summary_dataframe(group, EXPERIMENT="Exp1", DROP_SUBJECT_NUM=None, num_trials=80):
    def perc(metric,):
        return (metric / num_trials) * 100

    it = InitialThangs(EXPERIMENT)
    wins = perc(group.score_metrics.score_metric("wins")).flatten().tolist()
    indecisions = perc(group.score_metrics.score_metric("indecisions")).flatten().tolist()
    incorrects = perc(group.score_metrics.score_metric("incorrects")).flatten().tolist()
    correct_decisions = perc(group.movement_metrics.correct_initial_decisions).flatten().tolist()
    median_movement_onset_time = np.nanmedian(group.movement_metrics.movement_onset_times("task"), axis=2).flatten().tolist()
    mean_movement_onset_time = np.nanmean(group.movement_metrics.movement_onset_times("task"), axis=2).flatten().tolist()
    q1_median_movement_onset_time = np.nanquantile(group.movement_metrics.movement_onset_times("task"), 0.25, axis=2).flatten().tolist()
    q3_median_movement_onset_time = np.nanquantile(group.movement_metrics.movement_onset_times("task"), 0.75, axis=2).flatten().tolist()
    movement_onset_time_sd = np.nanstd(group.movement_metrics.movement_onset_times("task"), axis=2).flatten().tolist()
    gamble_movement_onset_time = np.nanmedian(group.react_guess_movement_metrics.movement_onset_times("react"), axis=2).flatten().tolist()
    median_movement_time = np.nanmedian(group.movement_metrics.movement_times("task"), axis=2).flatten().tolist()
    reaction_decisions = perc(group.react_guess_score_metrics.react_guess_decisions("react")).flatten().tolist()
    gamble_decisions = perc(group.react_guess_score_metrics.react_guess_decisions("guess")).flatten().tolist()
    wins_when_both_decide = group.score_metrics.wins_when_both_reach(perc=True).flatten().tolist()
    subject_number = np.repeat(np.arange(1, it.num_subjects + 1, 1, dtype=int), it.num_blocks).tolist()
    condition = np.tile(np.arange(1, it.num_blocks + 1, 1, dtype=int), it.num_subjects).tolist()
    # alt_condition = np.tile(['1000 (50)','1000 (150)','1100 (50)','1100 (150)', '1200 (50)', '1200 (150)'],it.num_subjects).tolist()
    if EXPERIMENT == "Exp1":
        factor1 = np.tile(["1000", "1000", "1100", "1100", "1200", "1200"], it.num_subjects)
        factor2 = np.tile(["50", "150"], it.num_subjects * 3)
        points = np.full_like(wins, 0)
    else:
        factor1 = np.tile(["0 Inc", "-1 Inc", "0 Inc", "-1 Inc"], it.num_subjects)
        factor2 = np.tile(["0 Ind", "0 Ind", "-1 Ind", "-1 Ind"], it.num_subjects)
        points = group.score_metrics.exp2_points_scored.flatten().tolist()

    df_metrics = pd.DataFrame(
        np.array(
            [
                mean_movement_onset_time,
                median_movement_onset_time,
                median_movement_time,
                wins,
                indecisions,
                incorrects,
                correct_decisions,
                wins_when_both_decide,
                gamble_movement_onset_time,
                movement_onset_time_sd,
                q1_median_movement_onset_time,
                q3_median_movement_onset_time,
                reaction_decisions,
                gamble_decisions,
                points,
            ]
        ).T,
        columns=[
            "Mean_Movement_Onset_Time",
            "Median_Movement_Onset_Time",
            "Median_Movement_Time",
            "Wins",
            "Indecisions",
            "Incorrects",
            "Correct_Decisions",
            "Wins_When_Both_Decide",
            "Median_Gamble_Movement_Onset_Time",
            "SD_Movement_Onset_Time",
            "Q1_Movement_Onset_Time",
            "Q3_Movement_onset_time",
            "Reaction_Decisions",
            "Guess_Decisions",
            "Points",
        ],
    )
    df_conditions = pd.DataFrame(np.array([subject_number, condition, factor1, factor2]).T, columns=["Subject", "Condition", "Factor_1", "Factor_2"])
    # df_metrics.astype('float64')

    # df = df.astype({'Subject':'int32','Condition':'int32','Condition Mean':'int32','Condition SD':'int32'})
    df = pd.concat([df_conditions, df_metrics], axis=1)
    df = df[df["Subject"].astype(int) != DROP_SUBJECT_NUM]
    # assert ~df.isnull().any(axis=1).any(),('NaN Values found in dataframe')
    print(f"!! DROPPING SUBJECT {DROP_SUBJECT_NUM} !! ")
    return df

def generate_exp1_trial_dataframe(group, EXPERIMENT="Exp1", DROP_SUBJECT_NUM=None, num_trials=80):
    '''
    Long format data of each subjects data for every trial (so not collapsed into a mean across trials)
    
    '''
    def perc(metric,):
        return (metric / num_trials) * 100
    
    def define_react_guess_decisions(
                                     player_movement_onset_times:np.ndarray, 
                                     agent_movement_onset_times:np.ndarray, 
                                     incorrect_mask, 
                                     subtract_off_reaction:int=25):
        mean_reaction_times = np.nanmedian(group.movement_metrics.reaction_times, axis=1)
        temp = mean_reaction_times - subtract_off_reaction
        adjusted_reaction_times = np.tile(temp,(it.num_blocks,num_trials))
        print(adjusted_reaction_times.shape)
        guess_mask =  (
            ((player_movement_onset_times - agent_movement_onset_times) <= adjusted_reaction_times) | (incorrect_mask)
        )
        return guess_mask, ~guess_mask
    

    it = InitialThangs(EXPERIMENT)
    win = group.score_metrics.win_mask.flatten().astype(int)
    indecision = group.score_metrics.indecision_mask.flatten().astype(int)
    incorrect = group.score_metrics.incorrect_mask.flatten().astype(int)
    movement_onset_times = group.movement_metrics.movement_onset_times("task").flatten()
    agent_movement_onset_times = group.raw_data.agent_task_leave_time.flatten()
    target_reach_times = group.movement_metrics.target_reach_times("task").flatten()
    movement_times = group.movement_metrics.movement_times("task").flatten()
    reaction_times = np.repeat(np.nanmedian(group.movement_metrics.reaction_times - 25, axis=1), it.num_blocks*it.num_trials) #! Subtracting off 25 ms
    timing_uncertainty = np.repeat(np.nanstd(group.raw_data.coincidence_reach_time, axis=1), it.num_blocks*it.num_trials)
    subject_number = np.repeat(np.arange(1, it.num_subjects + 1, 1, dtype=int), it.num_blocks*it.num_trials)
    condition = np.tile(np.repeat(np.arange(1, it.num_blocks + 1, 1, dtype=int),it.num_trials), it.num_subjects)
    trial_number = np.tile(np.arange(1,it.num_trials + 1, 1, dtype=int), it.num_subjects*it.num_blocks)
    # alt_condition = np.tile(['1000 (50)','1000 (150)','1100 (50)','1100 (150)', '1200 (50)', '1200 (150)'],it.num_subjects).tolist()
    if EXPERIMENT == "Exp1":
        factor1 = np.repeat(["1000", "1000", "1100", "1100", "1200", "1200"], it.num_subjects*it.num_trials)
        factor2 = np.repeat(["50","50","50", "150", "150","150"], it.num_subjects*it.num_trials)
        points = np.full_like(win, 0)
    else:
        factor1 = np.tile(["0 Inc", "-1 Inc", "0 Inc", "-1 Inc"], it.num_subjects)
        factor2 = np.tile(["0 Ind", "0 Ind", "-1 Ind", "-1 Ind"], it.num_subjects)
        points = group.score_metrics.exp2_points_scored.flatten().tolist()

    df_metrics = pd.DataFrame(
        np.array(
            [
                win,
                indecision,
                incorrect,
                movement_onset_times,
                agent_movement_onset_times,
                target_reach_times,
                movement_times,
                reaction_times,
                timing_uncertainty,
            ]
        ).T,
        columns=[
            "win",
            "indecision",
            "incorrect",
            "player_movement_onset_time",
            "agent_movement_onset_time",
            "target_reach_time",
            "movement_time",
            "reaction_time",
            "timing_uncertainty",
        ],
    )
    print(np.vstack((subject_number, condition, factor1, factor2)))
    df_conditions = pd.DataFrame(np.vstack((subject_number, condition, trial_number, factor1, factor2)).T, 
                                 columns=["Subject", "Condition","Trial","Factor_1", "Factor_2"])
    df = pd.concat([df_conditions, df_metrics], axis=1)
    df['reaction_decision'] = df["player_movement_onset_time"]-df["agent_movement_onset_time"]>=df['reaction_time']
    df['guess_decision'] = df["player_movement_onset_time"]-df["agent_movement_onset_time"]<df['reaction_time']
    return df

def generate_exp2_summary_dataframe(group2):
    it2 = InitialThangs("Exp2")

    react_mixed_median   = np.nanmedian(group2.movement_metrics.exp2_react_guess_reaction_time_split('react','mixed'),axis=1)
    react_only_median    = np.nanmedian(group2.movement_metrics.exp2_react_guess_reaction_time_split('react','only'),axis=1)
    guess_mixed_median   = np.nanmedian(group2.movement_metrics.exp2_react_guess_reaction_time_split('guess','mixed'),axis=1)
    guess_only_median    = np.nanmedian(group2.movement_metrics.exp2_react_guess_reaction_time_split('guess','only'),axis=1)
    reaction_time_median = np.vstack((react_mixed_median,guess_mixed_median,react_only_median,guess_only_median)).T.flatten()

    react_mixed_sd   = np.nanstd(group2.movement_metrics.exp2_react_guess_reaction_time_split('react','mixed'),axis=1)
    react_only_sd    = np.nanstd(group2.movement_metrics.exp2_react_guess_reaction_time_split('react','only'),axis=1)
    guess_mixed_sd   = np.nanstd(group2.movement_metrics.exp2_react_guess_reaction_time_split('guess','mixed'),axis=1)
    guess_only_sd    = np.nanstd(group2.movement_metrics.exp2_react_guess_reaction_time_split('guess','only'),axis=1)
    reaction_time_sd = np.vstack((react_mixed_sd,guess_mixed_sd,react_only_sd,guess_only_sd)).T.flatten()

    react_mixed_iqr   = iqr(group2.movement_metrics.exp2_react_guess_reaction_time_split('react','mixed'),axis=1,nan_policy='omit')
    react_only_iqr    = iqr(group2.movement_metrics.exp2_react_guess_reaction_time_split('react','only'),axis=1,nan_policy='omit')
    guess_mixed_iqr   = iqr(group2.movement_metrics.exp2_react_guess_reaction_time_split('guess','mixed'),axis=1,nan_policy='omit')
    guess_only_iqr    = iqr(group2.movement_metrics.exp2_react_guess_reaction_time_split('guess','only'),axis=1,nan_policy='omit')
    reaction_time_iqr = np.vstack((react_mixed_iqr,guess_mixed_iqr,react_only_iqr,guess_only_iqr)).T.flatten()

    subject_number = np.repeat(np.arange(1, it2.num_subjects + 1, 1, dtype=int), 4)
    condition = np.tile(np.arange(1, 4 + 1, 1, dtype=int), it2.num_subjects)    
    factor1 = np.tile(["React","Guess"], it2.num_subjects*2)
    factor2 = np.tile(["Mixed", "Mixed","Only","Only"], it2.num_subjects)

    df_metrics = pd.DataFrame(np.array([reaction_time_median,reaction_time_sd, reaction_time_iqr]).T,
                            columns=["Reaction_Time_Median","Reaction_Time_SD","Reaction_Time_IQR"])
    df_conditions = pd.DataFrame(np.array([subject_number, condition, factor1, factor2]).T, 
                                columns=["Subject", "Condition", "Factor_1", "Factor_2"])

    df2 = pd.concat([df_conditions, df_metrics], axis=1)
    return df2
    

def generate_exp2_trial_dataframe(group):
    '''
    THIS creates the raw data df, not the summary (like medians, sd, iqr)
    '''
    it = InitialThangs("Exp2")
    num_conditions = 3
    num_trials = 100
    subject_number = np.repeat(np.arange(1, it.num_subjects + 1, 1, dtype=int), num_conditions*num_trials) # Get subject 1 300 times, then sub 2 300 times
    condition = np.tile(np.repeat(['mixed','react_only','guess_only'],num_trials), it.num_subjects) # Get (1 100 times, then 2 100 times etc. and then tile that for each subject)
    reaction_decision_array = np.where(group.movement_metrics.reaction_decision_array.flatten()==1.0,'right','left')
    group.movement_metrics.get_reaction_times(filter_=False)
    reaction_time = group.movement_metrics.reaction_times.flatten() #! FILTERING REACTION TIMES <170 or >600 out in the subject object
    movement_time = group.movement_metrics.movement_times(task="reaction").flatten()
    decision_type = np.where(group.movement_metrics.reaction_guess_mask,"guess","react").flatten()
    df = pd.DataFrame({
        'subject':subject_number,"condition":condition,
        "decision_array":reaction_decision_array,
        "decision_type":decision_type,
        "reaction_time":reaction_time,
        "movement_time":movement_time,
                    
    })
    results_path = Path(r'D:\OneDrive - University of Delaware - o365\Desktop\MatchPennies-Agent-Expirement\results\participant_data')
    with open(results_path / "Exp2_reaction_time_df.pkl", "wb") as f:
        dill.dump(df,f)        
    
    return df



SAVE_PATH = Path(r"D:\OneDrive - University of Delaware - o365\Desktop\MatchPennies-Agent-Expirement\results\participant_data")
print('here')
# group = rdf.generate_subject_object_v3("Exp1", "All Trials", movement_metric_type='velocity')

group2 = rdf.generate_subject_object_v3("Exp2", "All Trials", movement_metric_type='velocity')

# df1 = generate_exp1_summary_dataframe(group, EXPERIMENT, DROP_SUBJECT_NUM=None)
df2 = generate_exp2_summary_dataframe(group2)

# Create raw dataframe if I want to use later for side analyses
# df3 = generate_exp1_trial_dataframe(group)
df4 = generate_exp2_trial_dataframe(group2)
# with open(SAVE_PATH / f"Exp1_summary_data_df.pkl", "wb") as f:
#     dill.dump(df1,f)
with open(SAVE_PATH / f"Exp2_summary_data_df.pkl", "wb") as f:
    dill.dump(df2,f)
# with open(SAVE_PATH / f"Exp1_trial_data_df.pkl", "wb") as f:
#     dill.dump(df3,f)
with open(SAVE_PATH / f"Exp2_trial_data_df.pkl", "wb") as f:
    dill.dump(df4,f)