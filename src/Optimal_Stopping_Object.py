import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import fftconvolve
import scipy.special as sc
import numba as nb
import numba_scipy  # Needs to be imported so that numba recognizes scipy (specificall scipy special erf)
import data_visualization as dv
from copy import deepcopy 
from numba_stats import norm
from functools import cached_property
from scipy import optimize
import time
import multiprocessing as mp
import loss_functions as lf
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
    shape = (time_sds.shape[0], time_means.shape[-1])
    EX_R, EX2_R, EX3_R = np.zeros((shape)), np.zeros((shape)), np.zeros((shape))
    EX_G, EX2_G, EX3_G = np.zeros((shape)), np.zeros((shape)), np.zeros((shape))
    dx = timesteps[1] - timesteps[0]

    for i in nb.prange(len(time_sds)):
        sig_y = time_sds[i]
        xpdf = agent_pdf[i, :]

        for j in range(time_means.shape[-1]):  # Need to loop through every possible decision time mean
            #*Commented out is the old way of doing this bc sc.erfc is recognized by numba, but now I know how to use norm.cdf with numba (which is the same as the error function)
            # xpdf = (1/(sig_x*np.sqrt(2*np.pi)))*np.e**((-0.5)*((timesteps - mu_x)/sig_x)**2) # Pdf of agent, used when getting expected value EX_R, etc.
            # prob_x_less_y = (sc.erfc((mu_x - mu_y[i])/(np.sqrt(2)*np.sqrt(sig_x**2 + sig_y**2))))/2 # Probability of a reaction decision, aka player decides after agent
            # y_integrated = np.empty(len(timesteps),dtype=np.float64)
            # y_inverse_integrated = np.empty(len(timesteps),dtype=np.float64)
            # for k in range(len(timesteps)): # Looping here bc numba_scipy version of sc.erfc can only take float, not an array
            #     t = timesteps[k]
            #     y_integrated[k] = (sc.erfc((t - mu_y[i])/(np.sqrt(2)*sig_y)))/2 # Going from x to infinity is the complementary error function (bc we want all the y's that are greater than x)
            #     y_inverse_integrated[k] = (sc.erfc((mu_y[i] - t)/(np.sqrt(2)*sig_y)))/2 # Swap limits of integration (mu_y[i] - t) now

            mu_y = time_means[j] # Put the timing mean in an easy to use variable,
            prob_x_less_y = prob_agent_less_player[i,j]  # get prob agent is less than player for that specific agent mean (i) and timing mean (j)
            prob_x_greater_y = 1 - prob_x_less_y
            y_integrated = 1 - norm.cdf(timesteps, mu_y, sig_y) # For ALL timesteps, what's the probabilit for every timing mean (from 0 to 2000) that the timing mean is greater than that current timestep
            y_inverse_integrated = 1 - y_integrated

            if prob_x_less_y != 0:
                EX_R[i, j] = nb_sum(timesteps*xpdf*y_integrated)*dx / prob_x_less_y
                EX2_R[i, j] = nb_sum((timesteps - EX_R[i, j]) ** 2*xpdf*y_integrated)*dx / prob_x_less_y  # SECOND CENTRAL MOMENT = VARIANCE
                # EX3_R[i,j] = 0 #np.sum((timesteps-EX_R[i,j])**3*xpdf*y_integrated)*dx/prob_x_less_y # THIRD CENTRAL MOMENT = SKEW
            else:
                EX_R[i, j] = 0
                EX2_R[i, j] = 0  # SECOND CENTRAL MOMENT = VARIANCE
                # EX3_R[i,j] = 0 # THIRD CENTRAL MOMENT = SKEW

            if prob_x_greater_y != 0:
                EX_G[i, j] = nb_sum(timesteps*xpdf*y_inverse_integrated)*dx / prob_x_greater_y
                EX2_G[i, j] = (
                    nb_sum((timesteps - EX_G[i, j]) ** 2*xpdf*y_inverse_integrated)*dx / prob_x_greater_y
                )  # SECOND CENTRAL MOMENT = VARIANCE
                # EX3_G[i,j] = 0#np.sum((timesteps-EX_G[i,j])**3*xpdf*y_inverse_integrated)*dx/prob_x_greater_y # THIRD CENTRAL MOMENT = SKEW
            else:
                EX_G[i, j] = 0
                EX2_G[i, j] = 0  # SECOND CENTRAL MOMENT = VARIANCE
                # EX3_G[i,j] = 0 # THIRD CENTRAL MOMENT = SKEW

    return EX_R, EX2_R, EX3_R, EX_G, EX2_G, EX3_G


def add_skewnorm_and_norm_distributions(timesteps,norm_mean,norm_sd,skewnorm_mean,skewnorm_sd,skewnorm_skew):
    '''
    The addition of skewnorm and norm will become essentially normal if the skew is small enough 
    
    But it is significant in terms of the new mean (20ms difference)
    '''
    dx = timesteps[0,0] - timesteps[0,1]
    norm_dis = stats.norm.pdf(timesteps, norm_mean, norm_sd)
    skewnorm_dis = stats.skewnorm.pdf(timesteps, skewnorm_skew, skewnorm_mean, skewnorm_sd)

    conv_pdf = fftconvolve(norm_dis,skewnorm_dis, mode = 'same')*dx
    mean = np.sum(timesteps*conv_pdf)*dx
    sd = np.sqrt(np.sum((timesteps-mean)**2*conv_pdf)*dx)
    skew = np.sqrt(np.sum((timesteps-mean)**3*conv_pdf)*dx)
    
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
    def __init__(self, **kwargs):
        """
        Model Inputs
        """
        #*Task Conditions
        if True:
            self.experiment = kwargs.get("experiment")
            self.num_blocks = kwargs.get("num_blocks")
            self.agent_means = kwargs.get("agent_means")  # If exp2, need to be np.array([1100]*4)
            self.agent_sds = kwargs.get("agent_sds")  # If exp2, need to be np.array([50]*4)
            self.nsteps = 1
            self.num_timesteps = kwargs.get("num_timesteps")
            self.timesteps = kwargs.get("timesteps", np.tile(np.arange(0.0, float(self.num_timesteps), self.nsteps), (2, self.num_blocks, 1)))
            self.old_timesteps = np.tile(np.arange(0.0, float(self.num_timesteps), self.nsteps), (self.num_blocks, 1))
            self.tiled_1500 = np.full_like(self.timesteps, 1500.0)
            self.tiled_agent_means = np.tile(self.agent_means, (self.timesteps.shape[-1], 1)).T
            self.tiled_agent_sds = np.tile(self.agent_sds, (self.timesteps.shape[-1], 1)).T

            self.neg_inf_cut_off_value = -100000
            # check = np.tile(np.arange(900.0, 1100.0, self.nsteps), (self.num_blocks, 1))
            # assert np.isclose(numba_cdf(check,np.array([5]*self.num_blocks), np.array([2]*self.num_blocks)),
            #                   stats.norm.cdf(check,np.array([5]),np.array([2]))).all()

        #*Player Parameters and Rewards
        if True:
            self.expected = kwargs.get("expected")
            if self.expected:
                self.key = 1 # 1 refers to 'exp' row
            else:
                self.key = 0 # 0 refers to 'true' row
    
            self.switch_cost_exists = kwargs.get('switch_cost_exists')
    
            #  HOW MUCH PEOPLE WEIGH WINS VERSUS CORRECTNESS IS THE BETA TERM
            self.prob_win_when_both_reach = kwargs.get("perc_wins_when_both_reach") / 100
            # self.BETA_ON                   = kwargs.get('BETA_ON')
            # self.BETA = self.find_beta_term()

            # Uncertainty
            self.reaction_sd = kwargs.get("reaction_sd")
            self.movement_sd = kwargs.get("movement_sd")
            self.timing_sd = kwargs.get("timing_sd")
            self.gamble_switch_sd = kwargs.get("gamble_switch_sd")
            self.electromechanical_sd = kwargs.get("electromechanical_sd")
            self.gamble_sd = self.gamble_switch_sd + self.electromechanical_sd + self.timing_sd

            self.reaction_reach_sd = np.sqrt(self.reaction_sd**2 + self.movement_sd**2)
            self.gamble_reach_sd = np.sqrt(self.gamble_sd**2 + self.movement_sd[:,np.newaxis]**2)

            # Ability
            self.reaction_time = kwargs.get("reaction_time")
            self.movement_time = kwargs.get("movement_time")
            self.reaction_plus_movement_time = self.reaction_time + self.movement_time
            
            self.gamble_switch_delay = kwargs.get("gamble_switch_delay")
            self.electromechanical_delay = kwargs.get('electromechanical_delay')
            self.gamble_delay = self.gamble_switch_delay + self.electromechanical_delay
            self.gamble_plus_movement_time = self.timesteps + self.movement_time[:,np.newaxis,np.newaxis] + self.gamble_delay[...,np.newaxis]

            assert self.electromechanical_delay[0] == self.electromechanical_delay[0]

            # Get reward matrix for Exp2
            if self.experiment == "Exp2":
                # Reward and cost values
                self.reward_matrix = kwargs.get("reward_matrix", np.array([[1, 0, 0], [1, -1, 0], [1, 0, -1], [1, -1, -1]]))
                self.condition_one = np.tile(self.reward_matrix[0], (1800, 1))
                self.condition_two = np.tile(self.reward_matrix[1], (1800, 1))
                self.condition_three = np.tile(self.reward_matrix[2], (1800, 1))
                self.condition_four = np.tile(self.reward_matrix[3], (1800, 1))
                
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

            # Ensure that if the switch cost doesn't exist, that the exp and true are the same
            assert self.switch_cost_exists is not None
            
            if not self.switch_cost_exists:
                assert self.gamble_switch_delay[1] == self.gamble_switch_delay[0] #! 0 refers to 'true' and 1 refers to 'exp'
                assert self.gamble_switch_sd[1] == self.gamble_switch_sd[0]
                assert np.sum(self.gamble_switch_delay + self.gamble_switch_sd) == 0
            # else:
            #     assert self.gamble_switch_delay[1] != self.gamble_switch_delay[0]
            #     assert self.gamble_switch_sd[1] != self.gamble_switch_sd[0]
            
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

        #*Get agent behavior
        self.cutoff_agent_behavior()

    @cached_property
    def prob_agent_has_gone(self):
        # temp = numba_cdf(self.inputs.timesteps,self.inputs.agent_means,self.inputs.agent_sds)
        temp = stats.norm.cdf(self.inputs.old_timesteps, self.inputs.agent_means[:,np.newaxis], self.inputs.agent_sds[:,np.newaxis])
        return temp

    @cached_property
    def prob_not_making(self):
        ans = 1 - stats.norm.cdf(1500,self.inputs.agent_means + 150,self.inputs.agent_sds)
        return ans[np.newaxis,:]
    
    @cached_property
    def prob_making(self):
        ans = stats.norm.cdf(1500,self.inputs.agent_means + 150,self.inputs.agent_sds)
        return ans[np.newaxis,:]
    
    @property
    def agent_moments(self):
        """
        Get first three central moments (EX2 is normalized for mean,
        EX3 is normalized for mean and sd) of the new distribution based on timing uncertainty
        
        IF I EVER USE TIMING_SD as something that could be not accounted for I'll have to fix this
        """
        #* Steps done outside for loop in get_moments to make it faster
        # Creates a 1,2000 inf timesteps, that can broadcast to 6,2000
        inf_timesteps = np.arange(0.0, 2000.0, self.inputs.nsteps)[np.newaxis,:]  # Going to 2000 is a good approximation, doesn't get better by going higher
        tiled_timing_sd = np.tile(self.inputs.timing_sd[self.inputs.key], (inf_timesteps.shape[-1], 1)).T  # Tile timing sd
        time_means = deepcopy(self.inputs.timesteps[0,0,:]) # Get the timing means that player can select as their stopping time
        agent_pdf = stats.norm.pdf(inf_timesteps, self.inputs.agent_means, self.inputs.agent_sds)  # Find agent pdf tiled 2000
        prob_agent_less_player = stats.norm.cdf(
            0, self.inputs.agent_means - inf_timesteps, np.sqrt(self.inputs.agent_sds**2 + (tiled_timing_sd) ** 2)
        )
        # Call get moments equation
        return get_moments(inf_timesteps.squeeze(), time_means.squeeze(), self.inputs.timing_sd[self.inputs.key,:], prob_agent_less_player, agent_pdf)

    def cutoff_agent_behavior(self):
        # Get the First Three moments for the left and right distributions (if X<Y and if X>Y respectively)
        EX_R, EX2_R, EX3_R, EX_G, EX2_G, EX3_G = self.agent_moments
        # no_inf_moments = [np.nan_to_num(x,nan=np.nan,posinf=np.nan,neginf=np.nan) for x in moments]

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

        assert np.allclose(self.prob_selecting_reaction + self.prob_selecting_gamble, 1.0)
        #*Leave times
        self.reaction_leave_time   = self.agent_behavior.reaction_leave_time + self.inputs.reaction_time[self.inputs.key] #! Keeping key here bc I don't plan on messing with reaction time expected versus gamble
        self.gamble_leave_time     = self.inputs.timesteps + self.inputs.gamble_delay[:,np.newaxis]
        self.wtd_leave_time = self.prob_selecting_reaction*self.reaction_leave_time + self.prob_selecting_gamble*self.gamble_leave_time
        #*Reach Times
        self.reaction_reach_time   = self.agent_behavior.reaction_leave_time + self.inputs.reaction_plus_movement_time[self.inputs.key]
        self.gamble_reach_time     = self.inputs.timesteps + self.inputs.gamble_delay[:,np.newaxis] + self.inputs.movement_time[:,np.newaxis,np.newaxis]
        self.wtd_reach_time = self.prob_selecting_reaction*self.reaction_reach_time + self.prob_selecting_gamble*self.gamble_reach_time
        #*Leave Time SD
        self.reaction_leave_time_sd = np.sqrt(self.agent_behavior.reaction_leave_time_sd**2 + self.inputs.reaction_sd[self.inputs.key] ** 2)
        #*If I pass an array, I took gamble leave time sd from the data
        if isinstance(self.inputs.gamble_sd[self.inputs.key], np.ndarray):
            self.gamble_leave_time_sd = self.inputs.gamble_sd[self.inputs.key][:, np.newaxis]
        else:  # If I didn't, then I need to throw on timing uncertainty and agent uncertainty to the decision sd
            self.gamble_leave_time_sd = np.sqrt(
                self.agent_behavior.gamble_leave_time_sd**2
                + self.inputs.gamble_sd[self.inputs.key] ** 2
                + tile(self.inputs.timing_sd[self.inputs.key] ** 2, self.inputs.num_timesteps)
            )
        self.wtd_leave_time_sd = (
            self.prob_selecting_reaction*self.reaction_leave_time_sd + self.prob_selecting_gamble*self.gamble_leave_time_sd
        )
        self.wtd_leave_time_iqr = (
            (self.wtd_leave_time + 0.675*self.wtd_leave_time_sd) - 
            (self.wtd_leave_time - 0.675*self.wtd_leave_time_sd)
        )
        #*Reach Time SD
        self.reaction_reach_time_sd = np.sqrt(self.reaction_leave_time_sd**2 + self.inputs.movement_sd[self.inputs.key] ** 2)
        self.gamble_reach_time_sd   = np.sqrt(self.gamble_leave_time_sd**2 + self.inputs.movement_sd[self.inputs.key] ** 2)
        self.wtd_reach_time_sd = (
            self.prob_selecting_reaction*self.reaction_reach_time_sd + self.prob_selecting_gamble*self.gamble_reach_time_sd
        )
        #*Predict Decision Time
        self.predicted_decision_time = (
            self.prob_selecting_reaction*self.agent_behavior.reaction_leave_time
            + self.prob_selecting_gamble*self.agent_behavior.gamble_leave_time
        )

    @property
    def prob_selecting_reaction(self):
        # Prob of SELECTING only includes timing uncertainty and agent uncertainty
        combined_sd = np.sqrt(
            self.inputs.timing_sd[self.inputs.key]**2 + self.inputs.agent_sds.squeeze()**2
        )  
        diff = self.inputs.timesteps - self.inputs.agent_means
        ans = 1 - stats.norm.cdf(0, diff, combined_sd[np.newaxis,:,np.newaxis])
        return ans

    @property
    def prob_selecting_gamble(self):
        return 1 - self.prob_selecting_reaction

    @property
    def prob_making_given_reaction(self):
        # Calculate the prob of making it on a reaction
        #! Cutoff agent distribution isn't normal, so might not be able to simply add these, problem for later
        mu = self.reaction_reach_time
        sd = np.sqrt(self.agent_behavior.reaction_leave_time_sd**2 + self.inputs.reaction_reach_sd[self.inputs.key] ** 2)
        # temp = numba_cdf(np.array([1500]),mu.flatten(),sd.flatten()).reshape(self.inputs.timesteps.shape)
        return stats.norm.cdf(1500, mu, sd)

    @property
    def prob_making_given_gamble(self):
        mu = self.gamble_reach_time
        sd = np.tile(self.inputs.gamble_reach_sd[self.inputs.key], (self.inputs.timesteps.shape[-1], 1)).T
        # temp = numba_cdf(np.array([1500]), mu.flatten(),sd.flatten()).reshape(self.inputs.timesteps.shape)
        return stats.norm.cdf(1500, mu, sd)


class ScoreMetrics:
    def __init__(self, model_inputs: ModelInputs, player_behavior: PlayerBehavior, agent_behavior: AgentBehavior):
        self.inputs = model_inputs
        #*These don't consider the probability that you select reaction
        # Prob of win
        self.prob_win_given_reaction = self.inputs.prob_selecting_correct_target_reaction*player_behavior.prob_making_given_reaction
        # Prob that you are correct and made it, OR you made it and the agent didn't, minus the probability of both of those things happening 
        # THIS INCLUDES THE AGENT NOT MAKING IT, will be basically 0 for everything but 1200,150 condition
        prob_win_given_gamble_if_agent_make = (
            self.inputs.prob_selecting_correct_target_gamble
            *player_behavior.prob_making_given_gamble
            *agent_behavior.prob_making
        )
        prob_win_given_gamble_if_agent_no_make = player_behavior.prob_making_given_gamble*agent_behavior.prob_not_making
        
        self.prob_win_given_gamble   = (
            prob_win_given_gamble_if_agent_make 
            + prob_win_given_gamble_if_agent_no_make 
            # - (prob_win_given_gamble_if_agent_make*prob_win_given_gamble_if_agent_no_make) #! Had to take this out for things to add up to 1, not sure why (6/23/23)
        )
        # Prob of incorrect
        self.prob_incorrect_given_reaction = (1 - self.inputs.prob_selecting_correct_target_reaction)*player_behavior.prob_making_given_reaction
        self.prob_incorrect_given_gamble   = (
            (1 - self.inputs.prob_selecting_correct_target_gamble)*player_behavior.prob_making_given_gamble*agent_behavior.prob_making 
        ) 

        # Prob of indecision
        self.prob_indecision_given_reaction = 1 - player_behavior.prob_making_given_reaction
        self.prob_indecision_given_gamble   = 1 - player_behavior.prob_making_given_gamble

        #*Prob making on reaction and gamble depends on the prob of selecting reaction and gamble too
        self.prob_making_reaction = player_behavior.prob_making_given_reaction*player_behavior.prob_selecting_reaction
        self.prob_making_gamble   = player_behavior.prob_making_given_gamble*player_behavior.prob_selecting_gamble
        self.prob_making          = self.prob_making_gamble + self.prob_making_reaction

        #*Multiply the actual probability of making it times the prob of getting it right for reaction and gamble
        self.prob_win_reaction = self.prob_win_given_reaction*player_behavior.prob_selecting_reaction
        self.prob_win_gamble   = self.prob_win_given_gamble*player_behavior.prob_selecting_gamble
        self.prob_win          = self.prob_win_reaction + self.prob_win_gamble

        #*Probability of receiving an incorrect cost
        self.prob_incorrect_reaction = self.prob_incorrect_given_reaction*player_behavior.prob_selecting_reaction
        self.prob_incorrect_gamble   = self.prob_incorrect_given_gamble*player_behavior.prob_selecting_gamble
        self.prob_incorrect          = self.prob_incorrect_reaction + self.prob_incorrect_gamble

        #*Probability of receiving an indecision cost (No chance of success for indecision, so we just multiply by the two marginal probs calculated in last section)
        self.prob_indecision_reaction = self.prob_indecision_given_reaction*player_behavior.prob_selecting_reaction
        self.prob_indecision_gamble   = self.prob_indecision_given_gamble*player_behavior.prob_selecting_gamble
        self.prob_indecision          = self.prob_indecision_reaction + self.prob_indecision_gamble

        self.correct_decisions = (
            player_behavior.prob_selecting_reaction*self.inputs.prob_selecting_correct_target_reaction
            + player_behavior.prob_selecting_gamble*self.inputs.prob_selecting_correct_target_gamble
        )

        assert np.allclose(self.prob_win + self.prob_incorrect + self.prob_indecision, 1.0)


class ExpectedReward:
    def __init__(self, model_inputs: ModelInputs, score_metrics: ScoreMetrics):
        self.inputs = model_inputs

        self.exp_reward_reaction = (
            score_metrics.prob_win_reaction*self.inputs.win_reward
            + score_metrics.prob_incorrect_reaction*self.inputs.incorrect_cost
            + score_metrics.prob_indecision_reaction*self.inputs.indecision_cost
        )

        self.exp_reward_gamble = (
            score_metrics.prob_win_gamble*self.inputs.win_reward
            + score_metrics.prob_incorrect_gamble*self.inputs.incorrect_cost
            + score_metrics.prob_indecision_gamble*self.inputs.indecision_cost
        )

        self.exp_reward = (
            score_metrics.prob_win*self.inputs.win_reward
            + score_metrics.prob_incorrect*self.inputs.incorrect_cost
            + score_metrics.prob_indecision*self.inputs.indecision_cost
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
        return np.nanargmax(self.er.exp_reward, axis=2).astype(int)

    @property
    def optimal_decision_time(self):
        return np.nanargmax(self.er.exp_reward, axis=2)*self.inputs.nsteps + np.min(self.inputs.timesteps)

    @property
    def optimal_exp_reward(self):
        return np.nanmax(self.er.exp_reward, axis=2)

    @property
    def fit_decision_time(self):
        return self.fit_decision_index*self.inputs.nsteps + np.min(self.inputs.timesteps)

    @property
    def fit_exp_reward(self):
        ans = np.zeros(self.inputs.num_blocks)
        for i in range(self.inputs.num_blocks):
            ans[i] = self.er.exp_reward[i, self.fit_decision_index[i]]
        return ans

    def set_fit_decision_index(self, index):
        self.fit_decision_index = index

    def get_metric(self, metric1, metric2=None, 
                   decision_type = 'optimal', metric_type='true'):
        '''
        The decision index will always choose from the self.inputs.key
          - MEANING that if the model EXPECTS no delays, then it'll use that decision time
          
        THEN it will apply that decision time onto the chosen metric_type array,
          - If the model is 'expected' then it should use the optimal decision type at the key associated with
            expectation (aka self.inputs.key = 1)
            - It will then apply that decision index onto the TRUE array
        '''
        
        if decision_type == "optimal":
            index = self.optimal_decision_index[self.inputs.key,:]
        elif decision_type == 'fit':
            index = self.fit_decision_index[self.inputs.key,:]
        else:
            raise ValueError("decision_type must be \"optimal\" or \"fit\"")
        
        if metric_type == 'true':
            metric_type_index = 0
        elif metric_type == 'expected':
            metric_type_index = 1
        else:
            raise ValueError('metric_type must be \'true\' or \'expected\'')
        
        if metric2 is None: # For none-reaction/gamble metrics
            ans = np.zeros(metric1.shape[1])*np.nan
            for i in range(metric1.shape[1]):
                ans[i] = metric1[metric_type_index, i, index[i]]
            return ans
        else: # For reaction/gamble metrics
            ans1 = np.zeros(metric1.shape[1])*np.nan
            ans2 = np.zeros(metric2.shape[1])*np.nan
            for i in range(metric1.shape[1]):
                ans1[i] = metric1[metric_type_index, i, index[i]]
                ans2[i] = metric2[metric_type_index, i, index[i]]
            return np.divide(ans1, ans2, out=np.zeros_like(ans2), where=ans2 > 1e-10)


class ModelConstructor:
    """
    Construct the model by sequentially creating each object, and passing it to the next object

    """

    def __init__(self, **kwargs):
        self.kwargs          = kwargs
        self.data_leave_times = kwargs.get('data_leave_times')
        self.inputs          = ModelInputs(**kwargs)
        self.agent_behavior  = AgentBehavior(self.inputs)
        self.player_behavior = PlayerBehavior(self.inputs, self.agent_behavior)
        self.score_metrics   = ScoreMetrics(self.inputs, self.player_behavior, self.agent_behavior)
        self.expected_reward = ExpectedReward(self.inputs, self.score_metrics)
        self.results         = Results(self.inputs, self.expected_reward)
        
    def fit_model(self, metric, target):
        '''
        Fitting the model with no free parameters
        - Just pick the decision time that minimizes the difference between model predicted movement onset
        and the data movement onset
        '''
        decision_index = np.array([500]*6)
        loss = np.zeros_like(self.inputs.timesteps)
        loss = abs(metric - target[:, np.newaxis])
        decision_index = np.argmin(loss, axis=1) + np.min(self.inputs.timesteps)
        self.results.set_fit_decision_index(decision_index.astype(int))


class ModelFitting:
    '''
    Pass ModelConstructor to fit free parameters
    
    1. Call run_model_fit_procedure, which calls get_loss from scipy
    2. get_loss calls update_model with free params supplied by scipy
    3. update_model runs through the model sequence with the new free parameters
    '''
    def __init__(self,model: ModelConstructor, fixed_decision_time=None):
        self.model = model
        # Use these two if I want to fit the unknown model switch delay, 
        # this fixes the optimal decision time and calculates things off of that
        self.fixed_decision_time = fixed_decision_time
        # Set the fit decision time
        if self.fixed_decision_time is not None:
            self.model.results.set_fit_decision_index(self.fixed_decision_time.astype(int))
            
        self.parameter_arr = []
        self.initial_param_shape         = None
        self.loss_store                  = None
        self.optimal_decision_time_store = None
        self.leave_time_store            = None
        self.leave_time_sd_store         = None
        
    def run_model_fit_procedure(self, free_params_init: dict, metric_keys: list, targets: np.ndarray,
                                method='Nelder-Mead', bnds=None, tol = 0.0000001, niter=100,
                                drop_condition_from_loss=None):
        self.loss_store = []
        self.optimal_decision_time_store = [] 
        self.leave_time_store = []
        self.leave_time_sd_store = []
        self.initial_guess = np.array(list(free_params_init.values())) # Get the free param values from dict and make an array, scipy will flatten it if it's 2D
        self.drop_condition_from_loss = drop_condition_from_loss
        num_params = len(self.initial_guess)
        if bnds is None:
            bnds = tuple([[0,500]])*num_params
        self.initial_param_shape = self.initial_guess.shape # Used for reshaping the parameter 
        rranges = tuple([slice(0,200,5)]*num_params)
        #* Select fit method
        if method=='brute':
            out = optimize.brute(self.get_loss, rranges,
                                 args=(metric_keys, targets, free_params_init.keys()),
                                 finish=None,full_output=True,
                                 ) 
            if not isinstance(out[0],np.ndarray):
                final_param_dict = dict(zip(free_params_init.keys(),[out[0]]))
            else:
                final_param_dict = dict(zip(free_params_init.keys(),out[0]))
                
        elif method=='basinhopping':
            out = optimize.basinhopping(self.get_loss, self.initial_guess,niter=niter,
                                 minimizer_kwargs={'method':'Nelder-Mead',
                                                   'args':(metric_keys, targets, free_params_init.keys())},
                                 stepsize=0.05
                                 )
            final_param_dict = dict(zip(free_params_init.keys(),out.x))
        else:
            out = optimize.minimize(self.get_loss, self.initial_guess, 
                                args=(metric_keys, targets, free_params_init.keys()), 
                                method=method, bounds=None, tol = tol)
            final_param_dict = dict(zip(free_params_init.keys(),out.x))
        
        #* Update model one last time
        self.update_model(final_param_dict)  
        self.parameter_arr               = np.array(self.parameter_arr)
        self.optimal_decision_time_store = np.array(self.optimal_decision_time_store)
        self.leave_time_store            = np.array(self.leave_time_store)
        self.leave_time_sd_store         = np.array(self.leave_time_sd_store)
        
        # ans = out.x + np.min(self.inputs.timesteps)
        # ans = out.x#.reshape(self.initial_param_shape)
        return out
    
    def get_loss(self, free_params_values, 
                 metric_keys, targets, 
                 free_params_keys, decision_type='optimal',):
        
        # Create dictionary back
        new_parameters_dict = dict(zip(free_params_keys,free_params_values))
        
        # If the standard deviation and mean combo reaches below 0, return a high loss
        if 'gamble_switch_delay' in free_params_keys and 'gamble_switch_sd' in free_params_keys:
            if new_parameters_dict['gamble_switch_delay'] - 2*new_parameters_dict['gamble_switch_sd']<0:
                return 1e3
        
        self.parameter_arr.append(free_params_values)
        # Get the new arrays from the optimized free parameter inputs
        self.update_model(new_parameters_dict) 
        # Get each metric from results at that specific decision time
        model_metrics = np.zeros_like(targets)
        for i in range(targets.shape[0]): 
            if 'leave_time' in metric_keys[i]:
                model_metric = getattr(self.model.player_behavior, metric_keys[i])
                # Find the metric at optimal decision time
                #! Metric type always being 'true' means that we use the decision_type for expected vs true, but the metric array
                #! We're using is ALWAYS the 'true' array. If we're fitting the true gamble delay, then the expected metric arrays shouldn't change
                model_metrics[i,:] = self.model.results.get_metric(model_metric, 
                                                                   decision_type=decision_type, 
                                                                    metric_type='true')  
            elif 'decision_time' in metric_keys[i]:
                model_metric = getattr(self.model.results,metric_keys[i])
                model_metrics[i,:] = model_metric
            else:
                model_metric = getattr(self.model.score_metrics, metric_keys[i])
                model_metrics[i,:] = self.model.results.get_metric(model_metric, 
                                                                   decision_type=decision_type, 
                                                                    metric_type='true')  # Find the metric at optimal decision time
        
        loss = lf.ape_loss(model_metrics, targets, drop_condition_num=self.drop_condition_from_loss)
        
        self.loss_store.append(loss)
        self.optimal_decision_time_store.append(self.model.results.optimal_decision_time[self.models.inputs.key]) # index at key bc I want the decision time for either expected or true
        self.leave_time_store.append(self.model.results.get_metric(self.model.player_behavior.wtd_leave_time,
                                                                   decision_type=decision_type,metric_type='true'))
        self.leave_time_sd_store.append(self.model.results.get_metric(self.model.player_behavior.wtd_leave_time_sd,
                                                                      decision_type=decision_type,metric_type='true'))
        
        return loss
    
    def update_model(self, free_param_dict):
        '''
        This updates the inputs using the new free parameters
        
        1. Update kwargs which updates model inputs
        2. Run through each step of the model with the new inputs
        3. Returns back to get_loss
        '''
        
        #* Change the keyword arguments that are being modified
        for k,v in free_param_dict.items():
            self.model.kwargs[k][0] = v #! Always changing the TRUE value, the expected value should always be set as of 8/15/23
        
        #* Pass new set of kwargs to the inputs, then run through model constructor sequence again
        self.model.inputs = ModelInputs(**self.model.kwargs) 
        
        #* Update Model
        if 'timing_sd' in free_param_dict.keys(): # AgentBehavior needs to be run again if the timing_sd changes
            print('AgentBehavior is being run again')
            self.model.agent_behavior = AgentBehavior(self.model.inputs)
            
        self.model.player_behavior = PlayerBehavior(self.model.inputs, self.model.agent_behavior)
        self.model.score_metrics   = ScoreMetrics(self.model.inputs, self.model.player_behavior, self.model.agent_behavior)
        self.model.expected_reward = ExpectedReward(self.model.inputs, self.model.score_metrics)
        self.model.results         = Results(self.model.inputs, self.model.expected_reward)
        
        #* Get new fit decision time (as of 6/26/23 I'm no longer fitting these decision times)
        # self.fit_model(self.player_behavior.wtd_leave_time,self.data_leave_times)

class Group_Models():
    def __init__(self, objects: dict, num_blocks: int, num_timesteps: float):
        self.objects = objects
        self.num_blocks = num_blocks
        self.num_timesteps = num_timesteps
        self.object_list = list(objects.values())
        self.num_subjects = len(self.object_list)

        # Get the sub-obj names (agent_behavior, player_behavior, etc. )
        sub_obj_names = [obj_name for obj_name in dir(self.object_list[0]) if not obj_name.startswith("__")]

        # Loop through the inner object names, and create an attribute for this group class
        # that returns a list of it's objects of those specific sub objects
        # Also, loop through eahc sub-objects attributes to get a list of those atttribute
        self.inner_object_attribute_names_dict = {}
        for sub_obj_name in sub_obj_names:
            # Set the sub-object names for the group to be the same, but this group returns a list of all the sub-objects
            setattr(self, sub_obj_name, [getattr(o, sub_obj_name) for o in self.object_list]) 
            # For every sub object name, get a dict of all the attributes it contains 
            self.inner_object_attribute_names_dict.update(
                {sub_obj_name: [attribute for attribute in dir(getattr(self.object_list[0], sub_obj_name)) if not attribute.startswith("__")]}
            )  

        # Get array of the optimal decision index and fit_decision_index for all the objects
        self.optimal_decision_index = np.array([o.results.optimal_decision_index for o in self.object_list])
        self.fit_decision_index = np.array([o.results.fit_decision_index for o in self.object_list])
    
    def get_input(self,metric_name,key='true'):
        return np.array([getattr(o,metric_name)[key] for o in self.inputs])
            
    def get_metric(self, object_name, metric_name, object_name2=None, metric_name2=None, metric_type="optimal"):      
        # Select optimal or fit decision index
        if metric_type == "optimal":
            indices = self.optimal_decision_index
        elif metric_type == "fit":
            indices = self.fit_decision_index
        
        ans = np.zeros((self.num_subjects, self.num_blocks))
        num = np.zeros((self.num_subjects, self.num_blocks))
        denom = np.zeros((self.num_subjects, self.num_blocks))
        # If it's not a reaction gamble metric
        if metric_name2 is None:
            # Get inner objects
            inner_objs = getattr(self, object_name)
            # Loop through all the inner objects 
            for i, inner_obj in enumerate(inner_objs):
                # Get the metric for each subject
                metric = getattr(inner_obj, metric_name)
                # Loop through blocks to return the metric at that optimal
                for j in range(self.num_blocks):
                    ans[i, j] = metric[j, indices[i, j]]
        else: 
            # Get inner objects
            inner_objs1 = getattr(self, object_name)
            inner_objs2 = getattr(self, object_name2)

            for i, (inner_obj1,inner_obj2) in enumerate(zip(inner_objs1,inner_objs2)):
                # Get the metric for each subject
                metric_num = getattr(inner_obj1, metric_name)
                metric_denom = getattr(inner_obj2, metric_name2)
                # Loop through blocks to return the metric at that optimal
                for j in range(self.num_blocks):
                    num[i, j] = metric_num[j, indices[i, j]]
                    denom[i, j] = metric_denom[j, indices[i, j]]
            ans = num/denom
        return ans

def main():
    m = ModelConstructor(
        experiment="Exp1",
        num_blocks=6,
        BETA_ON=False,
        agent_means=np.array([1000, 1000, 1100, 1100, 1200, 1200]).astype(float),
        agent_sds=np.array([100]*6).astype(float),
        reaction_time={"true": 275, "exp": 275},
        movement_time={"true": 150, "exp": 150},
        reaction_sd={"true": 25, "exp": 25},
        movement_sd={"true": 25, "exp": 25},
        timing_sd={"true": np.array([150]*6), "exp": np.array([150]*6)},
        perc_wins_when_both_reach=np.array([0.8]*6),
        gamble_sd={"true": 150, "exp": 10},
        gamble_delay={"true": 125, "exp": 50},
    )

    return


if __name__ == "__main__":
    main()


#################################################################################################
#################################################################################################
#################################################################################################
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
