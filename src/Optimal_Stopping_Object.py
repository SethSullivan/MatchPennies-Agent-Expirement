import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.special as sc
import numba as nb
import numba_scipy  # Needs to be imported so that numba recognizes scipy (specificall scipy special erf)
import data_visualization as dv
import copy
from numba_stats import norm
from functools import cached_property
from scipy import optimize
import time

wheel = dv.ColorWheel()
"""
04/04/23

v4 Takes away the Truncation stuff

04/17/23

Added in the flexibility to change reward around instead of agent mean and sd

"""


#####################################################
###### Helper Functions to get the Moments ##########
#####################################################
@nb.njit(fastmath=True)
def nb_sum(x):
    n_sum = 0
    for i in range(len(x)):
        n_sum += x[i]
    return n_sum


_ = nb_sum(np.array([2, 2]))


@nb.njit(parallel=True, fastmath=True)
def get_moments(timesteps, time_means, time_sds, prob_agent_less_player, agent_pdf):
    shape = (len(time_sds), len(time_means))
    EX_R, EX2_R, EX3_R = np.zeros((shape)), np.zeros((shape)), np.zeros((shape))
    EX_G, EX2_G, EX3_G = np.zeros((shape)), np.zeros((shape)), np.zeros((shape))
    dx = timesteps[1] - timesteps[0]

    for i in nb.prange(len(time_sds)):
        sig_y = time_sds[i]
        xpdf = agent_pdf[i, :]

        for j in range(len(time_means)):  # Need to loop through every possible decision time mean
            # * Commented out is the old way of doing this bc sc.erfc is recognized by numba, but now I know how to use norm.cdf with numba (which is the same as the error function)
            # xpdf = (1/(sig_x*np.sqrt(2*np.pi)))*np.e**((-0.5)*((timesteps - mu_x)/sig_x)**2) # Pdf of agent, used when getting expected value EX_R, etc.
            # prob_x_less_y = (sc.erfc((mu_x - mu_y[i])/(np.sqrt(2)*np.sqrt(sig_x**2 + sig_y**2))))/2 # Probability of a reaction decision, aka player decides after agent
            # y_integrated = np.empty(len(timesteps),dtype=np.float64)
            # y_inverse_integrated = np.empty(len(timesteps),dtype=np.float64)
            # for k in range(len(timesteps)): # Looping here bc numba_scipy version of sc.erfc can only take float, not an array
            #     t = timesteps[k]
            #     y_integrated[k] = (sc.erfc((t - mu_y[i])/(np.sqrt(2)*sig_y)))/2 # Going from x to infinity is the complementary error function (bc we want all the y's that are greater than x)
            #     y_inverse_integrated[k] = (sc.erfc((mu_y[i] - t)/(np.sqrt(2)*sig_y)))/2 # Swap limits of integration (mu_y[i] - t) now

            mu_y = time_means[j]  # Put the timing mean in an easy to use variable
            prob_x_less_y = prob_agent_less_player[i, j]  # get prob agent is less than player for that specific agent mean (i) and timing mean (j)
            prob_x_greater_y = 1 - prob_x_less_y
            y_integrated = 1 - norm.cdf(
                timesteps, mu_y, sig_y
            )  # For ALL timesteps, what's the probabilit for every timing mean (from 0 to 2000) that the timing mean is greater than that current timestep
            y_inverse_integrated = 1 - y_integrated

            if prob_x_less_y != 0:
                EX_R[i, j] = nb_sum(timesteps * xpdf * y_integrated) * dx / prob_x_less_y
                EX2_R[i, j] = nb_sum((timesteps - EX_R[i, j]) ** 2 * xpdf * y_integrated) * dx / prob_x_less_y  # SECOND CENTRAL MOMENT = VARIANCE
                # EX3_R[i,j] = 0 #np.sum((timesteps-EX_R[i,j])**3*xpdf*y_integrated)*dx/prob_x_less_y # THIRD CENTRAL MOMENT = SKEW
            else:
                EX_R[i, j] = 0
                EX2_R[i, j] = 0  # SECOND CENTRAL MOMENT = VARIANCE
                # EX3_R[i,j] = 0 # THIRD CENTRAL MOMENT = SKEW

            if prob_x_greater_y != 0:
                EX_G[i, j] = nb_sum(timesteps * xpdf * y_inverse_integrated) * dx / prob_x_greater_y
                EX2_G[i, j] = (
                    nb_sum((timesteps - EX_G[i, j]) ** 2 * xpdf * y_inverse_integrated) * dx / prob_x_greater_y
                )  # SECOND CENTRAL MOMENT = VARIANCE
                # EX3_G[i,j] = 0#np.sum((timesteps-EX_G[i,j])**3*xpdf*y_inverse_integrated)*dx/prob_x_greater_y # THIRD CENTRAL MOMENT = SKEW
            else:
                EX_G[i, j] = 0
                EX2_G[i, j] = 0  # SECOND CENTRAL MOMENT = VARIANCE
                # EX3_G[i,j] = 0 # THIRD CENTRAL MOMENT = SKEW

    return EX_R, EX2_R, EX3_R, EX_G, EX2_G, EX3_G


def numba_cdf(x, mu_arr, sig_arr):
    if x.ndim == 2:  # If x dim is 1, then we have the x as the (6,1800)
        assert x.shape[0] == mu_arr.shape[0]
        ans = np.zeros(x.shape)
        for i in range(len(mu_arr)):
            ans[i, :] = norm.cdf(x[i, :], mu_arr[i], sig_arr[i])
    else:  # Else, we have the mu_arr as the (6,1800) but we pass it in flattened
        ans = np.zeros(mu_arr.shape)
        for i in range(len(mu_arr)):
            ans[i] = norm.cdf(x, mu_arr[i], sig_arr[i])
    return ans


def add_dicts(d1, d2, d3=None):
    """
    Takes the keys from d1 and uses that in the new dictionary.
    Then adds the values from d1 and d2
    """
    if d3 is None:
        return {x1[0]: x1[1] + x2[1] for x1, x2 in zip(d1.items(), d2.items())}
    else:
        return {x1[0]: x1[1] + x2[1] + x3[1] for x1, x2, x3 in zip(d1.items(), d2.items(), d3.items())}


def combine_sd_dicts(d1, d2):
    d1_sq = {k: v**2 for k, v in d1.items()}
    d2_sq = {k: v**2 for k, v in d2.items()}
    temp = add_dicts(d1_sq, d2_sq)
    return {k: np.sqrt(v) for k, v in temp.items()}


def tile(arr, num):
    return np.tile(arr, (num, 1)).T


class ModelInputs:
    __slots__ = (
        "agent_means",
        "agent_sds",
        "condition_four",
        "condition_one",
        "condition_three",
        "condition_two",
        "experiment",
        "gamble_decision_sd",
        "gamble_delay",
        "gamble_reach_sd",
        "gamble_reach_time",
        "incorrect_cost",
        "indecision_cost",
        "movement_sd",
        "movement_time",
        "neg_inf_cut_off_value",
        "nsteps",
        "num_blocks",
        "num_timesteps",
        "prob_selecting_correct_target_gamble",
        "prob_selecting_correct_target_reaction",
        "prob_win_when_both_reach",
        "reaction_plus_movement_time",
        "reaction_reach_sd",
        "reaction_sd",
        "reaction_time",
        "reward_matrix",
        "tiled_1500",
        "tiled_agent_means",
        "tiled_agent_sds",
        "timesteps",
        "timesteps_dict",
        "timing_sd",
        "win_reward",
        "expected",
        "key",
    )

    def __init__(self, **kwargs):
        """
        Model Inputs
        """
        # * Task Conditions
        if True:
            self.experiment = kwargs.get("experiment")
            self.num_blocks = kwargs.get("num_blocks")
            self.agent_means = kwargs.get("agent_means")  # If exp2, need to be np.array([1100]*4)
            self.agent_sds = kwargs.get("agent_sds")  # If exp2, need to be np.array([50]*4)
            self.nsteps = 1
            self.num_timesteps = kwargs.get("num_timesteps")
            self.timesteps = kwargs.get("timesteps", np.tile(np.arange(0.0, float(self.num_timesteps), self.nsteps), (self.num_blocks, 1)))
            self.timesteps_dict = {"true": self.timesteps, "exp": self.timesteps}
            self.tiled_1500 = np.full_like(self.timesteps, 1500.0)
            self.tiled_agent_means = np.tile(self.agent_means, (self.timesteps.shape[-1], 1)).T
            self.tiled_agent_sds = np.tile(self.agent_sds, (self.timesteps.shape[-1], 1)).T

            self.neg_inf_cut_off_value = -100000
            check = np.tile(np.arange(900.0, 1100.0, self.nsteps), (self.num_blocks, 1))
            # assert np.isclose(numba_cdf(check,np.array([5]*self.num_blocks), np.array([2]*self.num_blocks)),
            #                   stats.norm.cdf(check,np.array([5]),np.array([2]))).all()

        # * Player Parameters and Rewards
        if True:
            self.expected = kwargs.get("expected")
            if self.expected:
                self.key = "exp"
            else:
                self.key = "true"
            #  HOW MUCH PEOPLE WEIGH WINS VERSUS CORRECTNESS IS THE BETA TERM
            self.prob_win_when_both_reach = kwargs.get("perc_wins_when_both_reach") / 100
            # self.BETA_ON                   = kwargs.get('BETA_ON')
            # self.BETA = self.find_beta_term()

            # Uncertainty
            self.reaction_sd = kwargs.get("reaction_sd")
            self.movement_sd = kwargs.get("movement_sd")
            self.timing_sd = kwargs.get("timing_sd")
            self.gamble_decision_sd = kwargs.get("gamble_decision_sd", {"true": np.array([50] * 6), "exp": np.array([10] * 6)})

            self.reaction_reach_sd = combine_sd_dicts(self.reaction_sd, self.movement_sd)
            self.gamble_reach_sd = combine_sd_dicts(self.gamble_decision_sd, self.movement_sd)

            # Ability
            self.reaction_time = kwargs.get("reaction_time")
            self.gamble_delay = kwargs.get("gamble_delay", {"true": np.array([150] * 6), "exp": np.array([50] * 6)})
            self.movement_time = kwargs.get("movement_time")
            self.reaction_plus_movement_time = add_dicts(self.reaction_time, self.movement_time)
            self.gamble_reach_time = add_dicts(self.timesteps_dict, self.movement_time, self.gamble_delay)

            # Reward and cost values
            self.reward_matrix = kwargs.get("reward_matrix", np.array([[1, 0, 0], [1, -1, 0], [1, 0, -1], [1, -1, -1]]))
            self.condition_one = np.tile(self.reward_matrix[0], (1800, 1))
            self.condition_two = np.tile(self.reward_matrix[1], (1800, 1))
            self.condition_three = np.tile(self.reward_matrix[2], (1800, 1))
            self.condition_four = np.tile(self.reward_matrix[3], (1800, 1))

            # Get reward matrix for Exp2
            if self.experiment == "Exp2":
                self.win_reward = np.vstack(
                    (self.condition_one[:, 0], self.condition_two[:, 0], self.condition_three[:, 0], self.condition_four[:, 0])
                )
                self.incorrect_cost = np.vstack(
                    (self.condition_one[:, 1], self.condition_two[:, 1], self.condition_three[:, 1], self.condition_four[:, 1])
                )
                self.indecision_cost = np.vstack(
                    (self.condition_one[:, 2], self.condition_two[:, 2], self.condition_three[:, 2], self.condition_four[:, 2])
                )
            else:
                self.win_reward = kwargs.get("win_reward", 1)
                self.incorrect_cost = kwargs.get("incorrect_cost", 0)
                self.indecision_cost = kwargs.get("indecision_cost", 0)
            # Prob of selecting the correct target
            self.prob_selecting_correct_target_reaction = kwargs.get("prob_selecting_correct_target_reaction", 1.0)
            self.prob_selecting_correct_target_gamble = kwargs.get("prob_selecting_correct_target_gamble", 0.5)


class AgentBehavior:
    def __init__(self, model_inputs: ModelInputs):
        self.inputs = model_inputs
        self.reaction_leave_time_var = None
        self.cutoff_reaction_skew = None
        self.reaction_leave_time = None
        self.reaction_leave_time_sd = None

        self.gamble_leave_time_var = None
        self.cutoff_gamble_skew = None
        self.gamble_leave_time = None
        self.gamble_leave_time_sd = None

        # * Get agent behavior
        self.cutoff_agent_behavior()

    @cached_property
    def prob_agent_has_gone(self):
        # temp = numba_cdf(self.inputs.timesteps,self.inputs.agent_means,self.inputs.agent_sds)
        temp = stats.norm.cdf(self.inputs.timesteps, self.inputs.agent_means, self.inputs.agent_sds)
        return temp

    @cached_property
    def agent_moments(self):
        """
        Get first three central moments (EX2 is normalized for mean,
        EX3 is normalized for mean and sd) of the new distribution based on timing uncertainty
        """
        inf_timesteps = np.arange(0.0, 2000.0, self.inputs.nsteps)  # Going to 2000 is a good approximation, doesn't get better by going higher
        inf_timesteps_tiled = np.tile(inf_timesteps, (self.inputs.num_blocks, 1))  # Tile to number of blocks
        inf_agent_means_tiled = np.tile(self.inputs.agent_means, (inf_timesteps.shape[0], 1)).T  # Tiled agents with 2000 timesteps
        inf_agent_sds_tiled = np.tile(self.inputs.agent_sds, (inf_timesteps.shape[0], 1)).T  # Tiled agetn sds with 2000 timesteps
        tiled_timing_sd = np.tile(self.inputs.timing_sd[self.inputs.key], (self.inputs.timesteps.shape[-1], 1)).T  # Tile timing sd
        time_means = self.inputs.timesteps[0, :]  # Get the timing means that player can select as their stopping time
        agent_pdf = stats.norm.pdf(inf_timesteps_tiled, inf_agent_means_tiled, inf_agent_sds_tiled)  # Find agent pdf tiled 2000
        prob_agent_less_player = stats.norm.cdf(
            0, self.inputs.tiled_agent_means - self.inputs.timesteps, np.sqrt(self.inputs.tiled_agent_sds**2 + (tiled_timing_sd) ** 2)
        )
        prob_player_less_agent = 1 - prob_agent_less_player  # Or do the same as above and swap mu_x and mu_y[j]
        # Call get moments equation
        return get_moments(inf_timesteps, time_means, self.inputs.timing_sd[self.inputs.key], prob_agent_less_player, agent_pdf)

    def cutoff_agent_behavior(self):
        # Get the First Three moments for the left and right distributions (if X<Y and if X>Y respectively)
        moments = self.agent_moments
        # no_inf_moments = [np.nan_to_num(x,nan=np.nan,posinf=np.nan,neginf=np.nan) for x in moments]
        EX_R, EX2_R, EX3_R, EX_G, EX2_G, EX3_G = moments

        self.reaction_leave_time, self.reaction_leave_time_var, self.cutoff_reaction_skew = EX_R, EX2_R, EX3_R
        self.reaction_leave_time_sd = np.sqrt(self.reaction_leave_time_var)

        self.gamble_leave_time, self.gamble_leave_time_var, self.cutoff_gamble_skew = EX_G, EX2_G, EX3_G
        self.gamble_leave_time_sd = np.sqrt(self.gamble_leave_time_var)


class PlayerBehavior:
    """
    This class contains the following for EVERY timestep

    1. Reaction/gamble leave/reach times and uncertainties
    2. Prob Selecting Reaction/Gamble
    3. Prob Making Given Reaction/Gamble
    """

    def __init__(self, model_inputs: ModelInputs, agent_behavior: AgentBehavior):
        self.inputs = model_inputs
        self.agent_behavior = agent_behavior

        if self.inputs.expected:
            self.key = "exp"
        else:
            self.key = "true"

        assert np.allclose(self.prob_selecting_reaction + self.prob_selecting_gamble, 1.0)
        # * Leave times
        self.reaction_leave_time = self.agent_behavior.reaction_leave_time + self.inputs.reaction_time[self.key]
        self.gamble_leave_time = self.inputs.timesteps + self.inputs.gamble_delay[self.key]
        self.wtd_leave_target_time = self.prob_selecting_reaction * self.reaction_leave_time + self.prob_selecting_gamble * self.gamble_leave_time
        # * Reach Times
        self.reaction_reach_time = self.agent_behavior.reaction_leave_time + self.inputs.reaction_plus_movement_time[self.key]
        self.gamble_reach_time = self.inputs.timesteps + self.inputs.gamble_delay[self.key] + self.inputs.movement_time[self.key]
        self.wtd_reach_target_time = self.prob_selecting_reaction * self.reaction_reach_time + self.prob_selecting_gamble * self.gamble_reach_time
        # * Leave Time SD
        self.reaction_leave_time_sd = np.sqrt(self.agent_behavior.reaction_leave_time_sd**2 + self.inputs.reaction_sd[self.key] ** 2)
        # * If I pass an array, I took gamble leave time sd from the data
        if isinstance(self.inputs.gamble_decision_sd[self.key], np.ndarray):
            self.gamble_leave_time_sd = self.inputs.gamble_decision_sd[self.key][:, np.newaxis]
            self.wtd_leave_target_time_sd = (
                self.prob_selecting_reaction * self.reaction_leave_time_sd + self.prob_selecting_gamble * self.gamble_leave_time_sd
            )

        else:  # If I didn't, then I need to throw on timing uncertainty and agent uncertainty to the decision sd
            self.gamble_leave_time_sd = np.sqrt(
                self.agent_behavior.gamble_leave_time_sd**2
                + self.inputs.gamble_decision_sd[self.key] ** 2
                + tile(self.inputs.timing_sd[self.key] ** 2, self.inputs.num_timesteps)
            )
            self.wtd_leave_target_time_sd = (
                self.prob_selecting_reaction * self.reaction_leave_time_sd + self.prob_selecting_gamble * self.gamble_leave_time_sd
            )
        # * Reach Time SD
        self.reaction_reach_time_sd = np.sqrt(self.reaction_leave_time_sd**2 + self.inputs.movement_sd[self.key] ** 2)
        self.gamble_reach_time_sd = np.sqrt(self.gamble_leave_time_sd**2 + self.inputs.movement_sd[self.key] ** 2)
        self.wtd_reach_target_time_sd = (
            self.prob_selecting_reaction * self.reaction_reach_time_sd + self.prob_selecting_gamble * self.gamble_reach_time_sd
        )
        # * Predict Decision Time
        self.predicted_decision_time = (
            self.prob_selecting_reaction * self.agent_behavior.reaction_leave_time
            + self.prob_selecting_gamble * self.agent_behavior.gamble_leave_time
        )

    @cached_property
    def prob_selecting_reaction(self):
        combined_sd = np.sqrt(
            self.inputs.timing_sd[self.key] ** 2 + self.inputs.agent_sds**2
        )  # Prob of SELECTING only includes timing uncertainty and agent uncertainty
        tiled_combined_sd = np.tile(combined_sd, (self.inputs.timesteps.shape[-1], 1)).T
        diff = self.inputs.timesteps - self.inputs.tiled_agent_means
        # ans = 1 - numba_cdf(np.array([0.0]),diff.flatten(),tiled_combined_sd.flatten()).reshape(self.inputs.timesteps.shape)
        ans = 1 - stats.norm.cdf(0, diff, tiled_combined_sd)
        return ans

    @cached_property
    def prob_selecting_gamble(self):
        return 1 - self.prob_selecting_reaction

    @cached_property
    def prob_making_given_reaction(self):
        # Calculate the prob of making it on a reaction
        #! Cutoff agent distribution isn't normal, so might not be able to simply add these, problem for later
        mu = self.reaction_reach_time
        sd = np.sqrt(self.agent_behavior.reaction_leave_time_sd**2 + self.inputs.reaction_reach_sd[self.key] ** 2)
        # temp = numba_cdf(np.array([1500]),mu.flatten(),sd.flatten()).reshape(self.inputs.timesteps.shape)
        return stats.norm.cdf(1500, mu, sd)

    @cached_property
    def prob_making_given_gamble(self):
        mu = self.gamble_reach_time
        sd = np.tile(self.inputs.gamble_reach_sd[self.key], (self.inputs.timesteps.shape[-1], 1)).T
        # temp = numba_cdf(np.array([1500]), mu.flatten(),sd.flatten()).reshape(self.inputs.timesteps.shape)
        return stats.norm.cdf(1500, mu, sd)


class ScoreMetrics:
    def __init__(self, model_inputs: ModelInputs, player_behavior: PlayerBehavior):
        self.inputs = model_inputs
        # * These don't consider the probability that you select reaction
        # Prob of win
        self.prob_win_given_reaction = self.inputs.prob_selecting_correct_target_reaction * player_behavior.prob_making_given_reaction
        self.prob_win_given_gamble = self.inputs.prob_selecting_correct_target_gamble * player_behavior.prob_making_given_gamble

        # Prob of incorrect
        self.prob_incorrect_given_reaction = (1 - self.inputs.prob_selecting_correct_target_reaction) * player_behavior.prob_making_given_reaction
        self.prob_incorrect_given_gamble = (1 - self.inputs.prob_selecting_correct_target_gamble) * player_behavior.prob_making_given_gamble

        # Prob of indecision
        self.prob_indecision_given_reaction = 1 - player_behavior.prob_making_given_reaction
        self.prob_indecision_given_gamble = 1 - player_behavior.prob_making_given_gamble

        # * Prob making on reaction and gamble depends on the prob of selecting reaction and gamble too
        self.prob_making_reaction = player_behavior.prob_making_given_reaction * player_behavior.prob_selecting_reaction
        self.prob_making_gamble = player_behavior.prob_making_given_gamble * player_behavior.prob_selecting_gamble
        self.prob_making = self.prob_making_gamble + self.prob_making_reaction

        # * Multiply the actual probability of making it times the prob of getting it right for reaction and gamble
        self.prob_win_reaction = self.prob_win_given_reaction * player_behavior.prob_selecting_reaction
        self.prob_win_gamble = self.prob_win_given_gamble * player_behavior.prob_selecting_gamble
        self.prob_win = self.prob_win_reaction + self.prob_win_gamble

        # * Probability of receiving an incorrect cost
        self.prob_incorrect_reaction = self.prob_incorrect_given_reaction * player_behavior.prob_selecting_reaction
        self.prob_incorrect_gamble = self.prob_incorrect_given_gamble * player_behavior.prob_selecting_gamble
        self.prob_incorrect = self.prob_incorrect_reaction + self.prob_incorrect_gamble

        # * Probability of receiving an indecision cost (No chance of success for indecision, so we just multiply by the two marginal probs calculated in last section)
        self.prob_indecision_reaction = self.prob_indecision_given_reaction * player_behavior.prob_selecting_reaction
        self.prob_indecision_gamble = self.prob_indecision_given_gamble * player_behavior.prob_selecting_gamble
        self.prob_indecision = self.prob_indecision_reaction + self.prob_indecision_gamble

        self.correct_decisions = (
            player_behavior.prob_selecting_reaction * self.inputs.prob_selecting_correct_target_reaction
            + player_behavior.prob_selecting_gamble * self.inputs.prob_selecting_correct_target_gamble
        )

        assert np.allclose(self.prob_win + self.prob_incorrect + self.prob_indecision, 1.0)


class ExpectedReward:
    def __init__(self, model_inputs: ModelInputs, score_metrics: ScoreMetrics):
        self.inputs = model_inputs

        self.exp_reward_reaction = (
            score_metrics.prob_win_reaction * self.inputs.win_reward
            + score_metrics.prob_incorrect_reaction * self.inputs.incorrect_cost
            + score_metrics.prob_indecision_reaction * self.inputs.indecision_cost
        )

        self.exp_reward_gamble = (
            score_metrics.prob_win_gamble * self.inputs.win_reward
            + score_metrics.prob_incorrect_gamble * self.inputs.incorrect_cost
            + score_metrics.prob_indecision_gamble * self.inputs.indecision_cost
        )

        self.exp_reward = (
            score_metrics.prob_win * self.inputs.win_reward
            + score_metrics.prob_incorrect * self.inputs.incorrect_cost
            + score_metrics.prob_indecision * self.inputs.indecision_cost
        )


class Results:
    """
    This class contains
    1. Find optimal function that uses the optimal index on the metrics calculated at every time step (From ScoreMetrics)
    2. Gets gamble/reaction calculations w/ first input being the gamble or reaction and second being the value that divides it
        - So we can get perc_reaction_wins which is (prob_win_reaction/prob_win)*100
    """

    def __init__(self, inputs: ModelInputs, er: ExpectedReward):
        self.inputs = inputs
        self.er = er
        self.fit_decision_index = None

    @property
    def optimal_decision_index(self):
        return np.nanargmax(self.er.exp_reward, axis=1).astype(int)

    @property
    def optimal_decision_time(self):
        return np.nanargmax(self.er.exp_reward, axis=1) * self.inputs.nsteps + np.min(self.inputs.timesteps)

    @property
    def optimal_exp_reward(self):
        return np.nanmax(self.er.exp_reward, axis=1)

    @property
    def fit_decision_time(self):
        return self.fit_decision_index * self.inputs.nsteps + np.min(self.inputs.timesteps)

    @property
    def fit_exp_reward(self):
        ans = np.zeros(self.inputs.num_blocks)
        for i in range(self.inputs.num_blocks):
            ans[i] = self.er.exp_reward[i, self.fit_decision_index[i]]
        return ans

    def set_fit_decision_index(self, index):
        self.fit_decision_index = index

    def get_metric(self, metric, metric_type="optimal"):
        if metric_type == "optimal":
            index = self.optimal_decision_index
        else:
            index = self.fit_decision_index

        ans = np.zeros(metric.shape[0]) * np.nan
        for i in range(metric.shape[0]):
            ans[i] = metric[i, index[i]]
        return ans

    def reaction_gamble_metric(self, metric1, metric2, metric_type="optimal"):
        """
        First metric is prob of that happening out of the second metric.
        np.divide handles the case where the denominator is 0 by just returning 0

        Example:
        Metric 1 = Prob Win Gamble
        Metric 2 = Prob Win
        Out      = Perc Wins That Were Gamble (Out of all the wins, how many were gambles)
        """
        arr1 = self.get_metric(metric1, metric_type=metric_type)
        arr2 = self.get_metric(metric2, metric_type=metric_type)
        return np.divide(arr1, arr2, out=np.zeros_like(arr2), where=arr2 > 1e-10) * 100


class ModelConstructor:
    """
    Construct the model by sequentially creating each object, and passing it to the next object

    """

    def __init__(self, **kwargs):
        self.inputs = ModelInputs(**kwargs)
        self.agent_behavior = AgentBehavior(self.inputs)
        self.player_behavior = PlayerBehavior(self.inputs, self.agent_behavior)
        self.score_metrics = ScoreMetrics(self.inputs, self.player_behavior)
        self.expected_reward = ExpectedReward(self.inputs, self.score_metrics)
        self.results = Results(self.inputs, self.expected_reward)

    def fit_model(self, metric, target):
        decision_index = np.array([500] * 6)
        loss = np.zeros_like(self.inputs.timesteps)
        loss = abs(metric - target[:, np.newaxis])
        decision_index = np.argmin(loss, axis=1) + np.min(self.inputs.timesteps)
        self.results.set_fit_decision_index(decision_index.astype(int))

    def fit_model_scipy(self, metric, target, init_decision_index: tuple):
        bnds = tuple([(np.min(self.inputs.timesteps), np.max(self.inputs.timesteps))] * self.inputs.num_blocks)
        ans = np.zeros((self.inputs.num_blocks))

        x = init_decision_index
        out = optimize.minimize(self.loss, x, args=(metric, target), method="Nelder-Mead", bounds=bnds)
        ans = out.x + np.min(self.inputs.timesteps)
        self.results.set_optimal_index(ans)

    def loss(self, decision_time, metric, target):
        decision_time = decision_time.astype(int)
        self.results.set_optimal_index(decision_time)  # Set the new optimal index
        model_metric = self.results.get_optimal(metric)  # Find the metric at that new optimal index
        return np.mean((model_metric - target) ** 2)


class Group_Models(ModelConstructor):
    def __init__(self, objects: dict, num_blocks: int, num_timesteps: float):
        self.objects = objects
        self.object_list = list(objects.values())
        self.num_subjects = len(self.object_list)
        self.num_blocks = num_blocks
        self.num_timesteps = num_timesteps

        # Get the sub-obj names (agent_behavior, player_behavior, etc. )
        sub_obj_names = [obj_name for obj_name in dir(self.object_list[0]) if not obj_name.startswith("__")]

        # Loop through the inner object names, and create an attribute for this group class
        # that returns a list of it's objects of those specific sub objects
        # Also, loop through eahc sub-objects attributes to get a list of those atttribute
        self.inner_object_attribute_names_dict = {}
        for sub_obj_name in sub_obj_names:
            # Set the sub-object names for the group to be the same, but this group returns a list of all the sub-objects
            setattr(
                self, sub_obj_name, [getattr(o, sub_obj_name) for o in self.object_list]
            ) 
            # For every sub object name, get a dict of all the attributes it contains 
            self.inner_object_attribute_names_dict.update(
                {sub_obj_name: [attribute for attribute in dir(getattr(self.object_list[0], sub_obj_name)) if not attribute.startswith("__")]}
            )  

        # Get array of the optimal decision index and fit_decision_index for all the objects
        self.optimal_decision_index = np.array([o.results.optimal_decision_index for o in self.object_list])
        self.fit_decision_index = np.array([o.results.fit_decision_index for o in self.object_list])

    def get(self, object_name, metric_name, metric_type="optimal"):
        if metric_type == "optimal":
            indices = self.optimal_decision_index
        elif metric_type == "fit":
            indices = self.fit_decision_index

        inner_objs = getattr(self, object_name)
        ans = np.zeros((self.num_subjects, self.num_blocks))
        for i, inner_obj in enumerate(inner_objs):
            metric = getattr(inner_obj, metric_name)
            for j in range(self.num_blocks):
                ans[i, j] = metric[j, indices[i, j]]
        return ans

def main():
    m = ModelConstructor(
        experiment="Exp1",
        num_blocks=6,
        BETA_ON=False,
        agent_means=np.array([1000, 1000, 1100, 1100, 1200, 1200]).astype(float),
        agent_sds=np.array([100] * 6).astype(float),
        reaction_time={"true": 275, "exp": 275},
        movement_time={"true": 150, "exp": 150},
        reaction_sd={"true": 25, "exp": 25},
        movement_sd={"true": 25, "exp": 25},
        timing_sd={"true": np.array([150] * 6), "exp": np.array([150] * 6)},
        perc_wins_when_both_reach=np.array([0.8] * 6),
        gamble_sd={"true": 150, "exp": 10},
        gamble_delay={"true": 125, "exp": 50},
    )

    return


if __name__ == "__main__":
    main()


#################################################################################################
#################################################################################################
###############################################################################################
@nb.njit(parallel=True)
def get_optimal_decision_time_for_certain_metric(ob, metric_name="RPMT"):
    """
    Trying to search across the entire space of reaction and movement times and find the optimal decision time for that person
    """
    # o = copy.deepcopy(ob)
    rts = np.arange(220, 400, 1, np.int64)
    mts = np.arange(100, 300, 1, np.int64)
    ans = np.zeros((len(rts), len(mts), ob.num_blocks))

    for i in nb.prange(len(rts)):
        for j in nb.prange(len(mts)):
            ob.reaction_time = rts[i]
            ob.movement_time = mts[j]
            ob.run_model()
            ans[i, j, :] = ob.optimal_decision_time
