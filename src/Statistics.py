import numpy as np
import pandas as pd
import analysis_utilities as au
import pingouin as pg
from itertools import combinations
import src.plot_functions as pf
import data_visualization as dv
import matplotlib.pyplot as plt
from initializer import InitialThangs
import read_data_functions as rdf
import copy

wheel = dv.ColorWheel()


# * Create dataframe function
def generate_dataframe(group, EXPERIMENT="Exp1", DROP_SUBJECT_NUM=13, num_trials=80):
    def perc(metric,):
        return (metric / num_trials) * 100

    it = InitialThangs(EXPERIMENT)
    wins = perc(group.score_metrics.score_metric("wins")).flatten().tolist()
    indecisions = perc(group.score_metrics.score_metric("indecisions")).flatten().tolist()
    incorrects = perc(group.score_metrics.score_metric("incorrects")).flatten().tolist()
    correct_decisions = perc(group.movement_metrics.correct_initial_decisions).flatten().tolist()
    median_movement_onset_time = np.nanmedian(group.movement_metrics.movement_onset_times("task"), axis=2).flatten().tolist()
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
        decision_time_difference_punish_incorrects = (
            np.nanmedian(group.movement_metrics.movement_onset_times("task"), axis=2)[0]
            - np.nanmedian(group.movement_metrics.movement_onset_times("task"), axis=2)[1]
        )
        decision_time_difference_punish_indecisions = (
            np.nanmedian(group.movement_metrics.movement_onset_times("task"), axis=2)[0]
            - np.nanmedian(group.movement_metrics.movement_onset_times("task"), axis=2)[2]
        )
    df_metrics = pd.DataFrame(
        np.array(
            [
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
            "Gamble_Decisions",
            "Points",
        ],
    )
    df_conditions = pd.DataFrame(np.array([subject_number, condition, factor1, factor2]).T, columns=["Subject", "Condition", "Factor_1", "Factor_2"])
    # df_metrics.astype('float64')

    # df = df.astype({'Subject':'int32','Condition':'int32','Condition Mean':'int32','Condition SD':'int32'})
    # dill.dump(df,open(save_path+'\\metrics_df_all_subjects.pkl','wb'))
    df = pd.concat([df_conditions, df_metrics], axis=1)
    df = df[df["Subject"].astype(int) != DROP_SUBJECT_NUM]
    # assert ~df.isnull().any(axis=1).any(),('NaN Values found in dataframe')
    print(f"!! DROPPING SUBJECT {DROP_SUBJECT_NUM} !! ")
    return df


def df_to_array(df_col, num_subjects, num_blocks):
    ans = np.array(df_col).reshape(num_subjects, num_blocks)
    return ans

def collapse_across(arr, combos):
            ans = []
            if len(combos[0]) == 2:
                for combo in combos:
                    a = int(combo[0])
                    b = int(combo[1])
                    ans.append(np.concatenate((arr[:, a], arr[:, b])))
            else:
                for combo in combos:
                    a = int(combo[0])
                    b = int(combo[1])
                    c = int(combo[2])
                    ans.append(np.concatenate((arr[:, a], arr[:, b], arr[:, c])))
            return_ans = np.array(ans).T
            assert return_ans.shape[0] > 15

            return return_ans
def get_combos(condition_nums, experiment):
    def _check_parity(combo):
        a = int(combo[0])
        b = int(combo[1])
        if a % 2 == b % 2:
            return True
        else:
            return False

    # * If not collapsing, then go through every combination
    if condition_nums is None:
        # But only want across the means, don't care about main effects of std
        if experiment == "Exp1":
            condition_nums = ["0", "1", "2", "3", "4", "5"]
            # Only take the even conditions together and the odd conditions together
            combos_ = [
                "".join(map(str, comb)) for comb in combinations(condition_nums, 2)
            ]  # Creates list of unique combos, order doesn't matter
            combos = [c for c in combos_ if _check_parity(c)]

        # Want every combo for exp2
        elif experiment == "Exp2":
            condition_nums = ["0", "1", "2", "3"]
            combos = [
                "".join(map(str, comb)) for comb in combinations(condition_nums, 2)
            ]  # Creates list of unique combos, order doesn't matter
    else:
        combos = ["".join(map(str, comb)) for comb in combinations(condition_nums, 2)]  # Creates list of unique combos, order doesn't matter
    return combos

class Inputs:
    FACTOR1 = "Factor_1"
    FACTOR2 = "Factor_2"

    def __init__(
        self, df, experiment, num_subjects, num_blocks, num_trials, trial_block_titles, 
        xlab, f1_xticklabels, f2_xticklabels, f1_xlabel, f2_xlabel, M
    ):
        self.experiment = experiment
        self.df = df
        self.num_subjects = num_subjects
        self.num_blocks = num_blocks
        self.num_trials = num_trials
        self.condition_nums = np.arange(0, self.num_blocks, 1).astype(str)
        self.trial_block_titles = trial_block_titles
        self.xlabel = xlab
        self.f1_xticklabels = f1_xticklabels
        self.f1_xlabel = f1_xlabel
        self.f2_xlabel = f2_xlabel
        self.f2_xticklabels = f2_xticklabels
        self.M_init = M
        assert self.experiment == "Exp1" or self.experiment == "Exp2"

        if self.experiment == "Exp1":
            self.f1_collapse_combos = ["01", "23", "45"]  # Combinations to run in pairwise bootstrap for MEAN factor
            self.f2_collapse_combos = ["024", "135"]  # Combinations to run in pairwise bootstrap for SD factor
            self.f1_condition_nums = ["0", "1", "2"]  # Call the condition numbers 0 1 2 for plotting
            self.f2_condition_nums = ["0", "1"]

        if self.experiment == "Exp2":
            self.f1_collapse_combos = [
                "02",
                "13",
            ]  # Combinations to run in pairwise bootstrap for INCORRECT PUNISHMENT factor, collapse across indecision factor
            self.f2_collapse_combos = [
                "01",
                "23",
            ]  # Combinations to run in pairwise bootstrap for INDECISION PUNISHMENT factor, collapse across incorrect factor
            self.f1_condition_nums = ["0", "1"]
            self.f2_condition_nums = ["0", "1"]


class Anova:
    def __init__(
        self,
        dependent_variable: str,
        inputs: Inputs,
        anova_type="rm_anova",
    ):
        self.inputs = inputs
        self.dependent_variable = dependent_variable
        self.anova_type = anova_type

    @property
    def anova(
        self,
    ):
        # Check to make sure that factor 1 is the means for exp1 or the incorrect punishment for exp2b or Reaction for exp2a
        assert (
            self.inputs.df[self.inputs.FACTOR1].str.contains("1000").any()
            or self.inputs.df[self.inputs.FACTOR1].str.contains("-1 Inc").any()
            or self.inputs.df[self.inputs.FACTOR1].str.contains("React").any()
        )
        if self.anova_type == "rm_anova":
            ans = pg.rm_anova(
                data=self.inputs.df, dv=self.dependent_variable, within=[self.inputs.FACTOR1, self.inputs.FACTOR2], 
                subject="Subject", detailed=True
            )
            assert ans["Source"][2] == f"{self.inputs.FACTOR1} * {self.inputs.FACTOR2}"  # Make sure row 2 is the interaction

        elif self.anova_type == "rm_manova":
            raise NotImplementedError("Still need to implement repeated measures MANOVA")

            # * Don't collapse
        return ans


class Bootstrap:
    def __init__(self, inputs: Inputs, anova_obj: Anova, change_m=None, alternative="two-sided", test="mean", no_collapse=False, **kwargs):
        self.inputs = inputs
        self.anova_obj = anova_obj
        if self.anova_obj is not None:
            self.anova = self.anova_obj.anova
            self.metric = df_to_array(self.inputs.df[self.anova_obj.dependent_variable], self.inputs.num_subjects, self.inputs.num_blocks)
        self.alternative = alternative
        self.test = test
        self.no_collapse = no_collapse
        self.collapse = None # Set in run_boostrap
        
        if change_m is None:
            self.M = copy.deepcopy(self.inputs.M_init)
        else:
            self.M = change_m
        
    def run_bootstrap(self):
        

        if self.anova["p-GG-corr"][2] < 0.05 or self.no_collapse:
            print("Significant interaction, doing pairwise bootstraps for each condition...")
            self.collapse = False
            pval_dict, cles_dict = self.pairwise_bootstrap(self.metric, condition_nums=self.inputs.condition_nums)
            return [pval_dict, cles_dict]
        # * Collapse
        else:
            print("Non-significant interaction, collapsing across conditions...")
            if self.anova["p-GG-corr"][2] < 0.1:
                print("Interaction significance close")
            self.collapse=True
            
            # * Factor 1 collapse
            f1_collapse_metric = collapse_across(self.metric, self.inputs.f1_collapse_combos)
            f1_collapse_pvals_dict, f1_collapse_cles_dict = self.pairwise_bootstrap(
                f1_collapse_metric, condition_nums=self.inputs.f1_condition_nums
            )

            # * Factor 2 collapse
            f2_collapse_metric = collapse_across(self.metric, self.inputs.f2_collapse_combos)
            f2_collapse_pvals_dict, f2_collapse_cles_dict = self.pairwise_bootstrap(
                f2_collapse_metric, condition_nums=self.inputs.f2_condition_nums
            )

            d = dict(
                zip(
                    ["f1pvals", "f1eff", "f2pvals", "f2eff"],
                    [f1_collapse_pvals_dict, f1_collapse_cles_dict, f2_collapse_pvals_dict, f2_collapse_cles_dict],
                )
            )

            return [f1_collapse_pvals_dict, f1_collapse_cles_dict, f2_collapse_pvals_dict, f2_collapse_cles_dict]

    def pairwise_bootstrap(self, data, condition_nums=None, **kwargs):

        def _get_alternative_dict(combos):
            if self.inputs.experiment == "Exp1":
                if self.alternative == "variable":
                    alternative_dict = {"02": "less", "04": "greater", "24": "greater", "13": "less", "15": "greater", "35": "greater"}
                else:
                    alternative_dict = dict(zip(combos, [self.alternative] * len(combos)))
            else:
                alternative_dict = dict(zip(combos, [self.alternative] * len(combos)))
            return alternative_dict
                
        # * Need these to be able to be generated incase I don't want to run an anova
        # This is a use case for the mini reaction time experiment
        if True:
            if not hasattr(self, "M"):
                self.M = kwargs.get("M")
            if not hasattr(self, "test"):
                self.test = kwargs.get("test")
            if not hasattr(self, "alternative"):
                self.alternative = kwargs.get('alternative')
            if not hasattr(self, "inputs"):
                self.alternative = kwargs.get('inputs')
                
        combos = get_combos(condition_nums, self.inputs.experiment)
        alternative_dict = _get_alternative_dict(combos)
        used_combos = alternative_dict.keys()
        if self.collapse:
            final_combos = combos
        else:
            final_combos = used_combos
        # if self.experiment == 'Exp1':
        pvals = {}
        cles1 = {}
        cles2 = {}
        c = -1
        for combo in final_combos:
            c += 1
            i = int(combo[0])
            j = int(combo[1])
            pvals.update({combo:au.bootstrap(data[:, i], data[:, j], paired=True, 
                                             M=self.M, alternative=alternative_dict[combo], 
                                             test=self.test)})
            cles1.update({combo:au.cles(data[:, i], data[:, j], paired=True)}) 
            cles2.update({combo:au.cles(data[:, j], data[:, i], paired=True)}) 

        # Create array and do holm bonferroni
        
        pvals_corrected = au.holmbonferroni_correction(np.array(list(pvals.values())))
        pvals_corrected_dict = dict(zip(used_combos,pvals_corrected))
        
        cles_dict = {}
        for (k1,v1),(k2,v2) in zip(cles1.items(),cles2.items()):
            if v1>v2:
                cles_dict.update({k1:v1})
            else:
                cles_dict.update({k2:v2})
        
        return pvals_corrected_dict, cles_dict

    def create_combos(self, num_list):
        return ["".join(map(str, comb)) for comb in combinations(num_list, 2)]
