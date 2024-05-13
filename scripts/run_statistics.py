import numpy as np
import pandas as pd 
import pingouin as pg 
import dill
from itertools import combinations
from pathlib import Path

from initializer import InitialThangs
import analysis_utilities as au

def df_to_array(df_col, num_subjects, num_blocks):
    ans = np.array(df_col).reshape(num_subjects, num_blocks)
    return ans

def exp1_collapse_across(arr, conditions:str):
    '''
    conditions: 'means' or 'sds'
    '''
    if conditions == "means":
        new_arr = np.vstack((
            np.concatenate((arr[:,0], arr[:,1])),
            np.concatenate((arr[:,2], arr[:,3])),
            np.concatenate((arr[:,4], arr[:,5]))
            )
        ).T
    else:
        new_arr = np.vstack((
            np.concatenate((arr[:,0], arr[:,2], arr[:,4])),
            np.concatenate((arr[:,1], arr[:,3], arr[:,5]))
            )
        ).T
    return new_arr

def exp2_collapse_across(arr, conditions:str):
    '''
    conditions: 'mixed_only' or 'react_guess'
    '''
    if conditions == "mixed_only":
        new_arr = np.vstack((
            np.concatenate((arr[:,0], arr[:,1])),
            np.concatenate((arr[:,2], arr[:,3])),
            )
        ).T
    elif conditions == "react_guess":
        new_arr = np.vstack((
            np.concatenate((arr[:,0], arr[:,2])),
            np.concatenate((arr[:,1], arr[:,3])),
            )
        ).T
    return new_arr

def check_parity(combo):
        a = int(combo[0])
        b = int(combo[1])
        if a % 2 == b % 2:
            return True
        else:
            return False
def get_combos(condition_nums=None, experiment="Exp1"):
    

    # * If not collapsing, then go through every combination
    if condition_nums is None:
        # But only want across the means, don't care about main effects of std
        if experiment == "Exp1":
            condition_nums = ["0", "1", "2", "3", "4", "5"]
            # Only take the even conditions together and the odd conditions together
            combos_ = [
                "".join(map(str, comb)) for comb in combinations(condition_nums, 2)
            ]  # Creates list of unique combos, order doesn't matter
            combos = [c for c in combos_ if check_parity(c)]

        # Want every combo for exp2
        elif experiment == "Exp2":
            condition_nums = ["0", "1", "2", "3"]
            combos = [
                "".join(map(str, comb)) for comb in combinations(condition_nums, 2)
            ]  # Creates list of unique combos, order doesn't matter
    else:
        combos = ["".join(map(str, comb)) for comb in combinations(condition_nums, 2)]  # Creates list of unique combos, order doesn't matter
    return combos

def pairwise_bootstrap(data, combos, alternative ="two-sided", M=1e6, test="mean", experiment="Exp1"):
    '''
    
    '''
    
    def _get_alternative_dict(combos):
        if experiment == "Exp1":
            if alternative == "variable":
                alternative_dict = {"02": "less", "04": "greater", "24": "greater", "13": "less", "15": "greater", "35": "greater"}
            else:
                alternative_dict = dict(zip(combos, [alternative] * len(combos)))
        else:
            alternative_dict = dict(zip(combos, [alternative] * len(combos)))
        #* Return one-liner filters the dictionary to only include the combos I've passed, necessary for variable alternative dict
        return {x: alternative_dict[x] for x in alternative_dict.keys() if x in combos}
            
    alternative_dict = _get_alternative_dict(combos)
    # if self.experiment == 'Exp1':
    pvals = {}
    cles1 = {}
    cles2 = {}
    c = -1
    for combo in combos:
        c += 1
        # print(combo)
        i = int(combo[0])
        j = int(combo[1])
        pvals.update({combo:au.bootstrap(data[:, i], data[:, j], paired=True, 
                                            M=M, alternative=alternative_dict[combo], 
                                            test=test)})
        cles1.update({combo:au.cles(data[:, i], data[:, j], paired=True)}) 
        cles2.update({combo:au.cles(data[:, j], data[:, i], paired=True)}) 
    
    cles_dict = {}
    for (k1,v1),(k2,v2) in zip(cles1.items(),cles2.items()):
        if v1>v2:
            cles_dict.update({k1:v1})
        else:
            cles_dict.update({k2:v2})
    
    return pvals, cles_dict


EXPERIMENT = "Exp1"
print(EXPERIMENT)

# Load dataframe of data to use for stats
with open(rf'D:\OneDrive - University of Delaware - o365\Desktop\MatchPennies-Agent-Expirement\results\participant_data\{EXPERIMENT}_summary_data_df.pkl', "rb") as f:
    df = dill.load(f)
it = InitialThangs(EXPERIMENT)
if EXPERIMENT == "Exp1":
    column_names = ["Mean_Movement_Onset_Time", "Median_Movement_Onset_Time", "SD_Movement_Onset_Time",
                    "Wins", "Incorrects", "Indecisions", "Guess_Decisions"]
    condition_nums = ["0", "1", "2", "3", "4", "5"]
    # Only care about mean comparisons, and then 01 and 12 for the collapse
    all_combos = ["01", "12", '02', '04', '13', '15', '24', '35']
    collapse_across = exp1_collapse_across
    collapse_arg = "means"
    collapsed_combos = ["01","02","12"]
    all_combo_slice = slice(2,None)
elif EXPERIMENT == "Exp2":
    column_names = ["Reaction_Time_Median", "Reaction_Time_SD","Reaction_Time_IQR"]
    condition_nums = ["0", "1", "2", "3",]
    all_combos = ["01","02","13","23"]
    collapse_across = exp2_collapse_across
    collapse_arg = "react_guess"
    collapsed_combos = ["01"]
    all_combo_slice = slice(0,None)
    
anova_df = pd.DataFrame()
pvals_df = pd.DataFrame(columns = ["Metric", "Collapsed"] + all_combos)
pvals_uncorr_df = pd.DataFrame(columns = ["Metric", "Collapsed"] + all_combos)
cles_df = pd.DataFrame(columns = ["Metric", "Collapsed"] + all_combos)

for i,colname in enumerate(column_names):
    print(colname)
    anova = pg.rm_anova(
                    data=df, dv=colname, 
                    within=["Factor_1", "Factor_2"], 
                    subject="Subject", detailed=True
                )
    
    anova.insert(0, "Metric", colname)
    anova_df = pd.concat([anova_df, anova])
    
    
    data_array = df_to_array(df[colname],
                             num_subjects=it.num_subjects, 
                             num_blocks=it.num_blocks)
    if anova["p-GG-corr"][2] < 0.05:
        print("Significant interaction, doing pairwise bootstraps for each condition...")
        new_combos = all_combos[all_combo_slice] # Only want to pairwise within same sds, so want 02,04,24, and 13,15,35
        collapsed=np.nan
    else: 
        print("Non-significant interaction, collapsing across conditions...")
        data_array = collapse_across(data_array,collapse_arg)
        new_combos = collapsed_combos
        collapsed = collapse_arg
    
    pvals,cles = pairwise_bootstrap(
        data = data_array,
        combos = new_combos,  
        alternative="two-sided",
        test='mean'
    )
    pvals_corrected = au.holmbonferroni_correction(list(pvals.values()))
    pvals_corrected = dict(zip(pvals.keys(),pvals_corrected))
    d1 = {"Metric":colname, "Collapsed":collapsed}
    df_row_pvals_uncorrected = d1 | pvals
    df_row_pvals = d1 | pvals_corrected 
    df_row_cles = d1 | cles
    pvals_uncorr_df = pd.concat([pvals_uncorr_df, pd.DataFrame(df_row_pvals_uncorrected,index=[i])]).fillna(np.nan)
    pvals_df = pd.concat([pvals_df, pd.DataFrame(df_row_pvals,index=[i])]).fillna(np.nan)
    cles_df = pd.concat([cles_df, pd.DataFrame(df_row_cles,index=[i])]).fillna(np.nan)

SAVE_PATH = Path(r"D:\OneDrive - University of Delaware - o365\Desktop\MatchPennies-Agent-Expirement\results\participant_data")

with open(SAVE_PATH / f"{EXPERIMENT}_pvals_uncorrected_df.pkl","wb") as f:
    dill.dump(pvals_uncorr_df, f)
    
with open(SAVE_PATH / f"{EXPERIMENT}_pvals_df.pkl","wb") as f:
    dill.dump(pvals_df, f)
    
with open(SAVE_PATH / f"{EXPERIMENT}_cles_df.pkl","wb") as f:
    dill.dump(cles_df, f)
    
with open(SAVE_PATH / f"{EXPERIMENT}_anova_df.pkl","wb") as f:
    dill.dump(anova_df, f)