from _pytest.junitxml import timing
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
# from functools import property
from functools import lru_cache
from scipy import optimize
import time
import multiprocessing as mp
import loss_functions as lf
import constants
import dill
wheel = dv.ColorWheel()
"""
Functions and Classes to generate and fit the optimal model
"""
#! Lookup Table for agent leave times as global constant, not in model inputs anymore
try:
    with open(constants.MODEL_INPUT_PATH / 'reaction_leave_time_lookup.pkl','rb') as f:
        agent_reaction_leave_time_lookup = dill.load(f)  
    with open(constants.MODEL_INPUT_PATH / 'reaction_leave_time_sd_lookup.pkl','rb') as f:
        agent_reaction_leave_time_sd_lookup = dill.load(f)  
    with open(constants.MODEL_INPUT_PATH / 'guess_leave_time_lookup.pkl','rb') as f:
        agent_guess_leave_time_lookup = dill.load(f)  
    with open(constants.MODEL_INPUT_PATH / 'guess_leave_time_sd_lookup.pkl','rb') as f:
        agent_guess_leave_time_sd_lookup = dill.load(f) 
except FileNotFoundError:
    print("Lookup Table not found, going to run through get_moments calculation")
    agent_reaction_leave_time_lookup = None
    agent_reaction_leave_time_sd_lookup = None
    agent_guess_leave_time_lookup = None
    agent_guess_leave_time_sd_lookup = None
    
#####################################################
###### Helper Functions to get the Moments ##########
#####################################################
@nb.njit(parallel=False,fastmath=True) # Parallel slows it down
def nb_sum(x):
    n_sum = 0
    for i in nb.prange(len(x)):
        n_sum += x[i]
    return n_sum
# Declaring actually slows it down
# @nb.njit(nb.types.UniTuple(nb.float64[:,:,:],6)(nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:,:]),
#          parallel=True, fastmath=True)
@nb.njit(parallel=True, fastmath=False) # For some reason fastmath slows it down a bit
def get_moments(timesteps:np.ndarray, time_means:np.ndarray, time_sds:np.ndarray, 
                prob_agent_less_player:np.ndarray, agent_pdf:np.ndarray):
    '''
    timesteps: (2000,)
    time_means: (1800,)
    time_sds: (2,6)
    prob_agent_less_player: (2,6,2000)
    agent_pdf: (2,6,2000)
    '''
    shape = (time_sds.shape[0], time_sds.shape[1], time_means.shape[-1])
    EX_R, EX2_R = np.zeros((shape)), np.zeros((shape)) 
    EX_G, EX2_G = np.zeros((shape)), np.zeros((shape))
    dx = timesteps[1] - timesteps[0]

    #* Looping expected/unexpected
    for i in range(time_sds.shape[0]):
        #* looping over conditions
        for j in range(time_sds.shape[1]):
            #* 11/02/23 - Removed these so numba doesn't have any temporary arrays
            # sig_y = time_sds[i,j]
            # xpdf = agent_pdf[i,j,:]
            #* Looping over possible mean decision times
            for k in nb.prange(time_means.shape[0]):
                # mu_y = time_means[k] # Put the timing mean in an easy to use variable,
                y_integrated = 1 - norm.cdf(timesteps, time_means[k], time_sds[i,j]) # For ALL timesteps, what's the probabilit for every timing mean (from 0 to 2000) that the timing mean is greater than that current timestep
                # y_inverse_integrated = 1 - y_integrated
                if prob_agent_less_player[i,j,k]!=0:
                    # integral_sum_R = nb_sum(timesteps*agent_pdf[i,j,:]*y_integrated)*dx
                    EX_R[i,j,k] = nb_sum(timesteps*agent_pdf[i,j,:]*y_integrated)*dx/prob_agent_less_player[i,j,k]
                    # SECOND CENTRAL MOMENT = VARIANCE
                    # integral_sum2_R = nb_sum((timesteps - EX_R[i,j,k])**2*agent_pdf[i,j,:]*y_integrated)*dx
                    EX2_R[i,j,k] = nb_sum((timesteps - EX_R[i,j,k])**2*agent_pdf[i,j,:]*y_integrated)*dx/prob_agent_less_player[i,j,k]  # SECOND CENTRAL MOMENT = VARIANCE
                    # EX3_R[i,j] = 0 #np.sum((timesteps-EX_R[i,j])**3*agent_pdf[i,j,:]*y_integrated)*dx/prob_agent_less_player[i,j,k] # THIRD CENTRAL MOMENT = SKEW
                else:
                    EX_R[i,j,k] = 0
                    EX2_R[i,j,k] = 0
                    
                if (1-prob_agent_less_player[i,j,k])!=0:
                    # integral_sum_G = nb_sum(timesteps*agent_pdf[i,j,:]*(1 - y_integrated))*dx
                    EX_G[i,j,k] =  nb_sum(timesteps*agent_pdf[i,j,:]*(1 - y_integrated))*dx / (1-prob_agent_less_player[i,j,k])
                    # integral_sum2_G = nb_sum((timesteps - EX_G[i,j,k])**2*agent_pdf[i,j,:]*(1 - y_integrated))*dx
                    EX2_G[i,j,k] = nb_sum((timesteps - EX_G[i,j,k])**2*agent_pdf[i,j,:]*(1 - y_integrated))*dx / (1-prob_agent_less_player[i,j,k])
                    # EX3_G[i,j] = 0#np.sum((timesteps-EX_G[i,j])**3*agent_pdf[i,j,:]*(1 - y_integrated))*dx/prob_x_greater_y # THIRD CENTRAL MOMENT = SKEW
                else:
                    EX_G[i,j,k] = 0
                    EX2_G[i,j,k] = 0
                    
    return EX_R, EX2_R, EX_G, EX2_G


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


class ModelInputs:
    def __init__(self, **kwargs):
        self.reset(**kwargs)
        
    def reset(self,**kwargs):
        #*Task Conditions
        if True:
            self.experiment = kwargs.get("experiment")
            self.num_blocks = kwargs.get("num_blocks")
            self.agent_means = kwargs.get("agent_means")  # If exp2, need to be np.array([1100]*4)
            self.agent_sds = kwargs.get("agent_sds") # If exp2, need to be np.array([50]*4)
            self.nsteps = kwargs.get('nsteps',1)
            self.num_timesteps = int(kwargs.get("num_timesteps")/self.nsteps)
            self.timesteps = kwargs.get("timesteps", np.tile(np.arange(0.0, float(self.num_timesteps), self.nsteps), 
                                                             (2, self.num_blocks, 1)
                                                             )
                                        ) # Has shape starting with (2,)
            self.round_num = kwargs.get('round_num',20)

        #*Player Parameters and Rewards
        if True:
            #* Expected sets the key that determines which array will be used when calculating the optimal. 
            #* This decision time is then applied onto the true array
            # ! To modulate which values aren't and are accounted for, need to make [0,...] == [1,...]
            self.expected = kwargs.get("expected")
            if self.expected:
                self.key = 1 # 1 refers to 'exp' row
            else:
                self.key = 0 # 0 refers to 'true' row
            self.use_agent_behavior_lookup = kwargs.get("use_agent_behavior_lookup",True)
            
            # Uncertainty
            self.reaction_sd     = kwargs.get("reaction_sd")
            self.movement_sd     = kwargs.get("movement_sd")
            self.timing_sd       = kwargs.get("timing_sd")
            self.guess_switch_sd = kwargs.get("guess_switch_sd")
            self.guess_sd        = kwargs.get("guess_sd") #! OPTION to directly use guess leave time sd
            self.use_true_guess_sd = kwargs.get("use_true_guess_sd",False)
            self.electromechanical_sd = kwargs.get("electromechanical_sd",0)
            # If i don't directly use data, then guess_sd is the combination of timing_sd (includes electromechanical sd probably) and guess_switch_sd
            if self.guess_sd is None or not self.use_true_guess_sd:
                self.guess_sd_from_data = False
                self.guess_sd = np.sqrt(self.guess_switch_sd**2 + self.timing_sd**2 + self.electromechanical_sd**2)
            else:
                self.guess_sd_from_data = True
                self.guess_sd = self.guess_sd

            # Ability
            self.reaction_time = kwargs.get("reaction_time")
            self.movement_time = kwargs.get("movement_time")
            
            self.guess_switch_delay       = kwargs.get("guess_switch_delay")
            self.electromechanical_delay  = kwargs.get('electromechanical_delay')
            self.guess_delay              = self.guess_switch_delay + self.electromechanical_delay

            assert self.electromechanical_delay[0] == self.electromechanical_delay[0]

            # Get reward matrix for Exp2
            if self.experiment == "Exp2":
                # Reward and cost values
                # This uses the base reward matrix and if incorrect or indecision are altered, it's added on to the base
                input_win_reward = kwargs.get("win_reward", 1)
                input_incorrect_cost = kwargs.get("incorrect_cost", 0)
                input_indecision_cost = kwargs.get("indecision_cost", 0)
                if isinstance(input_win_reward, np.ndarray):
                    self.win_reward = input_win_reward
                    self.incorrect_cost = input_incorrect_cost
                    self.indecision_cost = input_indecision_cost
                else:
                    reward_matrix = np.array([[1, 0, 0], [1, -1, 0], [1, 0, -1], [1, -1, -1]])
                    reward_change_arr = np.array([input_win_reward, input_incorrect_cost, input_indecision_cost])
                    condition_one = np.tile(reward_matrix[0], (self.num_timesteps, 1))
                    condition_two = np.tile(reward_matrix[1], (self.num_timesteps, 1))
                    condition_three = np.tile(reward_matrix[2], (self.num_timesteps, 1))
                    condition_four = np.tile(reward_matrix[3], (self.num_timesteps, 1))
                    
                    win_reward_temp = np.vstack(
                        (condition_one[:, 0], condition_two[:, 0], condition_three[:, 0], condition_four[:, 0])
                    )
                    incorrect_cost_temp = np.vstack(
                        (condition_one[:, 1], condition_two[:, 1], condition_three[:, 1], condition_four[:, 1])
                    )
                    indecision_cost_temp = np.vstack(
                        (condition_one[:, 2], condition_two[:, 2], condition_three[:, 2], condition_four[:, 2])
                    )
                    
                    self.win_reward = win_reward_temp # Don't want to change this, generally
                    self.incorrect_cost = incorrect_cost_temp + input_incorrect_cost
                    self.indecision_cost = indecision_cost_temp + input_indecision_cost
                
                assert np.all(self.win_reward>=1)
                
            else:
                self.win_reward = kwargs.get("win_reward", 1)
                self.incorrect_cost = kwargs.get("incorrect_cost", 0)
                self.indecision_cost = kwargs.get("indecision_cost", 0)
            # Prob of selecting the correct target
            self.prob_selecting_correct_target_reaction = kwargs.get("prob_selecting_correct_target_reaction", 1.0)
            self.prob_selecting_correct_target_guess = kwargs.get("prob_selecting_correct_target_guess", 0.5)
            
class AgentBehavior:
    def __init__(self, model_inputs: ModelInputs):
        self.reset(model_inputs)
        
    def reset(self,model_inputs):
        self.inputs = model_inputs
        self.reaction_leave_time_var = None
        self.cutoff_reaction_skew = None
        self.reaction_leave_time = None
        self.reaction_leave_time_sd = None

        self.guess_leave_time_var = None
        self.cutoff_guess_skew = None
        self.guess_leave_time = None
        self.guess_leave_time_sd = None

        #*Get agent behavior
        self.cutoff_agent_behavior()

    @property
    def prob_agent_has_gone(self):
        # temp = numba_cdf(self.inputs.timesteps,self.inputs.agent_means,self.inputs.agent_sds)
        temp = stats.norm.cdf(self.inputs.timesteps[0], self.inputs.agent_means, self.inputs.agent_sds)
        
        return temp

    @property
    def prob_not_making(self):
        ans = 1 - stats.norm.cdf(1500,self.inputs.agent_means + 150,self.inputs.agent_sds)
        return ans
    
    @property
    def prob_making(self):
        ans = stats.norm.cdf(1500,self.inputs.agent_means + 150,self.inputs.agent_sds)
        return ans
    
     # Only need to run this function if timing_sd changes, or agent_means/agent_sds change (don't need to worry about them tho)
    def agent_moments(self):
        """
        Get first three central moments (EX2 is normalized for mean,
        EX3 is normalized for mean and sd) of the new distribution based on timing uncertainty
        """
        #* Steps done outside for loop in get_moments to make it faster
        #
        # DOing this bc lru cache needs a hashable data type (aka a float or int)
        # We then recreate the timing_sd array which is just one number in (2,6,1) shape
        timing_sd = self.inputs.timing_sd
            
        # Creates a 1,1,2000 inf timesteps, that can broadcast to 2,6,1
        inf_timesteps = np.arange(0.0, 2000.0, self.inputs.nsteps)[np.newaxis,np.newaxis,:] # Going to 2000 is a good approximation, doesn't get better by going higher
        time_means = deepcopy(self.inputs.timesteps[0,0,:]) # Get the timing means that player can select as their stopping time
        agent_pdf = stats.norm.pdf(inf_timesteps, self.inputs.agent_means, self.inputs.agent_sds)  # Find agent pdf
        prob_agent_less_player = stats.norm.cdf(
            0, self.inputs.agent_means - inf_timesteps, 
            np.sqrt(self.inputs.agent_sds**2 + (timing_sd) ** 2)
        )
        #* Returns tuple of First three moments for left (Reaction) and right (Gamble) of distribution
        #* Don't actually use the skew at all
        return_vals = get_moments(
            inf_timesteps.squeeze(), 
            time_means.squeeze(), 
            timing_sd.squeeze(), # Squeezing for numba
            prob_agent_less_player.squeeze(), 
            agent_pdf.squeeze()
        )
        return return_vals

    def cutoff_agent_behavior(self):
        # Get the First Three moments for the left and right distributions (if X<Y and if X>Y respectively)
        if self.inputs.use_agent_behavior_lookup:
            try:
                self.reaction_leave_time = agent_reaction_leave_time_lookup[int(self.inputs.timing_sd[0,0,0]),int(self.inputs.timing_sd[1,0,0])]
                self.reaction_leave_time_sd = agent_reaction_leave_time_sd_lookup[int(self.inputs.timing_sd[0,0,0]),int(self.inputs.timing_sd[1,0,0])]
                self.guess_leave_time = agent_guess_leave_time_lookup[int(self.inputs.timing_sd[0,0,0]),int(self.inputs.timing_sd[1,0,0])]
                self.guess_leave_time_sd = agent_guess_leave_time_sd_lookup[int(self.inputs.timing_sd[0,0,0]),int(self.inputs.timing_sd[1,0,0])]          
            except IndexError:
                EX_R, EX2_R, EX_G, EX2_G = self.agent_moments()
                self.reaction_leave_time, self.reaction_leave_time_var = EX_R, EX2_R
                self.reaction_leave_time_sd = np.sqrt(self.reaction_leave_time_var)

                self.guess_leave_time, self.guess_leave_time_var, = EX_G, EX2_G
                self.guess_leave_time_sd = np.sqrt(self.guess_leave_time_var)
        else:
            EX_R, EX2_R, EX_G, EX2_G = self.agent_moments()
            self.reaction_leave_time, self.reaction_leave_time_var = EX_R, EX2_R
            self.reaction_leave_time_sd = np.sqrt(self.reaction_leave_time_var)

            self.guess_leave_time, self.guess_leave_time_var, = EX_G, EX2_G
            self.guess_leave_time_sd = np.sqrt(self.guess_leave_time_var)

class PlayerBehavior:
    """
    This class contains the following for EVERY timestep

    1. Reaction/guess leave/reach times and uncertainties
    2. Prob Selecting Reaction/guess
    3. Prob Making Given Reaction/guess
    """

    def __init__(self, model_inputs: ModelInputs, agent_behavior: AgentBehavior):
        self.reset(model_inputs,agent_behavior)
        
    def reset(self, model_inputs,agent_behavior):
        self.inputs = model_inputs
        self.agent_behavior = agent_behavior

        assert np.allclose(self.prob_selecting_reaction + self.prob_selecting_guess, 1.0)
        #*Leave times
        self.reaction_leave_time   = self.agent_behavior.reaction_leave_time + self.inputs.reaction_time
        self.guess_leave_time     = self.inputs.timesteps + self.inputs.guess_delay
        self.wtd_leave_time = self.prob_selecting_reaction*self.reaction_leave_time + self.prob_selecting_guess*self.guess_leave_time
        #*Reach Times
        self.reaction_reach_time   = self.agent_behavior.reaction_leave_time + self.inputs.reaction_time + self.inputs.movement_time
        self.guess_reach_time     = self.inputs.timesteps + self.inputs.guess_delay + self.inputs.movement_time
        self.wtd_reach_time = self.prob_selecting_reaction*self.reaction_reach_time + self.prob_selecting_guess*self.guess_reach_time
        #*Leave Time SD
        self.reaction_leave_time_sd = np.sqrt(self.agent_behavior.reaction_leave_time_sd**2 
                                              + self.inputs.reaction_sd ** 2)
        
        #! NOT SURE IF AGENT BEHAVIOR SHOULD INFLUENCE THIS, (8/16/23 i say it should bc it looks like guess leave time sd changes for 1000 and 1100 conditions btwn 50 and 150)
        # Also, the model predicts high guess switch sd when not accounting for it in order to get the best fit. This doesn't seem reflective of reality
        #! 11/1/23 - I don't think it should. Doesn't make sense with how I've modeled this. The guess time sd is only your timing sd, electro sd, and switch sd
        if self.inputs.guess_sd_from_data: 
            self.guess_leave_time_sd = self.inputs.guess_sd # DOESN'T Add on agent behavior, took it from data which includes that
        else:
            # self.guess_leave_time_sd = np.sqrt(self.agent_behavior.guess_leave_time_sd**2 + # Does include agent behavior bc it might effect , guess_sd includes timing from inputs
            #                                     self.inputs.guess_sd**2
            #                             )
            self.guess_leave_time_sd = self.inputs.guess_sd
        
        self.wtd_leave_time_sd = (
            self.prob_selecting_reaction*self.reaction_leave_time_sd + self.prob_selecting_guess*self.guess_leave_time_sd
        )
        self.wtd_leave_time_iqr = (
            (self.wtd_leave_time + 0.675*self.wtd_leave_time_sd) - 
            (self.wtd_leave_time - 0.675*self.wtd_leave_time_sd)
        )
        #*Reach Time SD
        self.reaction_reach_time_sd = np.sqrt(self.reaction_leave_time_sd**2 + self.inputs.movement_sd ** 2)
        self.guess_reach_time_sd   = np.sqrt(self.guess_leave_time_sd**2 + self.inputs.movement_sd** 2) # NEed to put these nweaxis in so that it's added on the first axis (aka expected veruss true axis) 
        self.wtd_reach_time_sd = (
            self.prob_selecting_reaction*self.reaction_reach_time_sd + self.prob_selecting_guess*self.guess_reach_time_sd
        )
        
        assert self.guess_reach_time_sd.shape == self.guess_leave_time_sd.shape
        assert self.reaction_reach_time_sd.shape == self.reaction_leave_time_sd.shape
        
        #*Predict Decision Time
        self.predicted_decision_time = (
            self.prob_selecting_reaction*self.agent_behavior.reaction_leave_time
            + self.prob_selecting_guess*self.agent_behavior.guess_leave_time
        )

    @property
    def prob_selecting_reaction(self):
        # Prob of SELECTING only includes timing uncertainty and agent uncertainty
        combined_sd = np.sqrt(
            self.inputs.timing_sd**2 + self.inputs.agent_sds**2
        )
        diff = self.inputs.timesteps - self.inputs.agent_means
        ans = 1 - stats.norm.cdf(0, diff, combined_sd)
        return ans

    @property
    def prob_selecting_guess(self):
        return 1 - self.prob_selecting_reaction

    @property
    def prob_making_given_reaction(self):
        # Calculate the prob of making it on a reaction
        #! Cutoff agent distribution isn't normal, so might not be able to simply add these, problem for later
        mu = self.reaction_reach_time
        sd = np.sqrt(self.agent_behavior.reaction_leave_time_sd**2 + self.reaction_reach_time_sd ** 2)
        # temp = numba_cdf(np.array([1500]),mu.flatten(),sd.flatten()).reshape(self.inputs.timesteps.shape)
        return stats.norm.cdf(1500, mu, sd)

    @property
    def prob_making_given_guess(self):
        mu = self.guess_reach_time
        sd = self.guess_reach_time_sd
        # temp = numba_cdf(np.array([1500]), mu.flatten(),sd.flatten()).reshape(self.inputs.timesteps.shape)
        return stats.norm.cdf(1500, mu, sd)


class ScoreMetrics:
    def __init__(self, model_inputs: ModelInputs, player_behavior: PlayerBehavior, agent_behavior: AgentBehavior):
        self.reset(model_inputs,player_behavior,agent_behavior)
        
    def reset(self,model_inputs,player_behavior,agent_behavior):
        self.inputs = model_inputs
        #*These don't consider the probability that you select reaction
        # Prob of win
        self.prob_win_given_reaction = self.inputs.prob_selecting_correct_target_reaction*player_behavior.prob_making_given_reaction
        # Prob that you are correct and made it, OR you made it and the agent didn't, minus the probability of both of those things happening 
        # THIS INCLUDES THE AGENT NOT MAKING IT, will be basically 0 for everything but 1200,150 condition
        prob_win_given_guess_if_agent_make = (
            self.inputs.prob_selecting_correct_target_guess
            *player_behavior.prob_making_given_guess
            *agent_behavior.prob_making
        )
        prob_win_given_guess_if_agent_no_make = player_behavior.prob_making_given_guess*agent_behavior.prob_not_making
        
        self.prob_win_given_guess   = (
            prob_win_given_guess_if_agent_make 
            + prob_win_given_guess_if_agent_no_make 
            # - (prob_win_given_guess_if_agent_make*prob_win_given_guess_if_agent_no_make) #! Had to take this out for things to add up to 1, not sure why (6/23/23)
        )
        # Prob of incorrect
        self.prob_incorrect_given_reaction = (1 - self.inputs.prob_selecting_correct_target_reaction)*player_behavior.prob_making_given_reaction
        self.prob_incorrect_given_guess   = (
            (1 - self.inputs.prob_selecting_correct_target_guess)*player_behavior.prob_making_given_guess*agent_behavior.prob_making 
        ) 

        # Prob of indecision
        self.prob_indecision_given_reaction = 1 - player_behavior.prob_making_given_reaction
        self.prob_indecision_given_guess   = 1 - player_behavior.prob_making_given_guess

        #*Prob making on reaction and guess depends on the prob of selecting reaction and guess too
        self.prob_making_reaction = player_behavior.prob_making_given_reaction*player_behavior.prob_selecting_reaction
        self.prob_making_guess   = player_behavior.prob_making_given_guess*player_behavior.prob_selecting_guess
        self.prob_making          = self.prob_making_guess + self.prob_making_reaction

        #*Multiply the actual probability of making it times the prob of getting it right for reaction and guess
        self.prob_win_reaction = self.prob_win_given_reaction*player_behavior.prob_selecting_reaction
        self.prob_win_guess   = self.prob_win_given_guess*player_behavior.prob_selecting_guess
        self.prob_win          = self.prob_win_reaction + self.prob_win_guess

        #*Probability of receiving an incorrect cost
        self.prob_incorrect_reaction = self.prob_incorrect_given_reaction*player_behavior.prob_selecting_reaction
        self.prob_incorrect_guess   = self.prob_incorrect_given_guess*player_behavior.prob_selecting_guess
        self.prob_incorrect          = self.prob_incorrect_reaction + self.prob_incorrect_guess

        #*Probability of receiving an indecision cost (No chance of success for indecision, so we just multiply by the two marginal probs calculated in last section)
        self.prob_indecision_reaction = self.prob_indecision_given_reaction*player_behavior.prob_selecting_reaction
        self.prob_indecision_guess   = self.prob_indecision_given_guess*player_behavior.prob_selecting_guess
        self.prob_indecision          = self.prob_indecision_reaction + self.prob_indecision_guess

        self.correct_decisions = (
            player_behavior.prob_selecting_reaction*self.inputs.prob_selecting_correct_target_reaction
            + player_behavior.prob_selecting_guess*self.inputs.prob_selecting_correct_target_guess
        )

        assert np.allclose(self.prob_win + self.prob_incorrect + self.prob_indecision, 1.0)
        
        # self.suboptimal_wtd_exp_reward = self.inputs.alpha*self.exp_reward + (1 - self.inputs.alpha)*self. 

class Results:
    """
    This class contains
    1. Find optimal function that uses the optimal index on the metrics calculated at every time step (From ScoreMetrics)
    2. Gets guess/reaction calculations w/ first input being the guess or reaction and second being the value that divides it
        - So we can get perc_reaction_wins which is (prob_win_reaction/prob_win)*100
    """

    def __init__(self, inputs: ModelInputs,score_metrics: ScoreMetrics):
        self.reset(inputs,score_metrics)
        
    def reset(self,inputs,score_metrics):
        self.inputs = inputs
        self.score_metrics = score_metrics
        self.fit_decision_index = None
        # self.max_exp_reward = np.nanmax(np.round(self.results.exp_reward,3),axis=2)
        # self.last_max_index = np.argwhere(np.round(self.results.exp_reward,3) - self.max_exp_reward == 0)[-1]
    
    @property
    def exp_reward_reaction(self):
        ans = (
            self.score_metrics.prob_win_reaction*self.inputs.win_reward
            + self.score_metrics.prob_incorrect_reaction*self.inputs.incorrect_cost
            + self.score_metrics.prob_indecision_reaction*self.inputs.indecision_cost
        )
        return ans
    @property
    def exp_reward_guess(self):
        ans = (
            self.score_metrics.prob_win_guess*self.inputs.win_reward
            + self.score_metrics.prob_incorrect_guess*self.inputs.incorrect_cost
            + self.score_metrics.prob_indecision_guess*self.inputs.indecision_cost
        )
        return ans
    
    @property
    def exp_reward(self):
        ans = np.round(
            self.score_metrics.prob_win*self.inputs.win_reward
            + self.score_metrics.prob_incorrect*self.inputs.incorrect_cost
            + self.score_metrics.prob_indecision*self.inputs.indecision_cost,
            self.inputs.round_num
        )
        return ans
        
        
    @property
    def optimal_decision_index(self):
        #* Not rounded, forward argmax
        # return np.nanargmax(self.exp_reward, axis=2).astype(int)*self.inputs.nsteps
        #* Rounded, forward
        # return np.nanargmax(np.round(self.exp_reward,self.inputs.round_num), axis=2).astype(int)*self.inputs.nsteps
        #* Rounded, backward
        a = np.round(self.exp_reward,self.inputs.round_num)
        return a.shape[2] - 1 - np.argmax(np.flip(a, axis=2), axis=2)

    @property
    def optimal_decision_time(self):
        # return np.nanargmax(self.exp_reward, axis=2)*self.inputs.nsteps + np.min(self.inputs.timesteps)
        return self.optimal_decision_index*self.inputs.nsteps + np.min(self.inputs.timesteps)

    @property
    def optimal_exp_reward(self):
        # return np.nanmax(self.exp_reward, axis=2)
        # return np.nanmax(np.round(self.exp_reward,self.inputs.round_num), axis=2)
        #* Rounded, backward
        a = np.round(self.exp_reward,self.inputs.round_num)
        return np.max(a,axis=2)

    @property
    def fit_decision_time(self):
        return self.fit_decision_index*self.inputs.nsteps + np.min(self.inputs.timesteps)

    @property
    def fit_exp_reward(self):
        ans = np.zeros(self.inputs.num_blocks)
        rounded_exp_reward = np.round(self.exp_reward,self.inputs.round_num)
        for i in range(self.inputs.num_blocks):
            ans[i] = rounded_exp_reward[i, self.fit_decision_index[i]]
        return ans

    def set_fit_decision_index(self, index):
        self.fit_decision_index = index

    def get_metric(self, metric1, metric2=None, 
                   decision_type = 'optimal', 
                   metric_type='true', key = None):
        '''
        
        decision_type = "optimal" or "fit"
        metric_type   = "true" or "expected"
        
        The decision index will always choose from the self.inputs.key
          - MEANING that if the model EXPECTS no delays, then it'll use that decision time
          
        THEN it will apply that decision time onto the chosen metric_type array,
          - If the model is 'expected' then it should use the optimal decision type at the key associated with
            expectation (aka self.inputs.key = 1)
            - It will then apply that decision index onto the TRUE array
        '''
        metric1 = metric1.squeeze()
        if metric2 is not None:
            metric2 = metric2.squeeze()
        
        #* This uses the optimal decision time of expected versus true
        if key is None:
            key = self.inputs.key
        
        #    - BC we want to use the optimal decision times from the EXPECTED arrays and apply those onto the TRUE arrays for unknown case
        if decision_type == "optimal":
            timing_index = self.optimal_decision_index[key,:] # Need self inputs key so that if it's expected, we take the optimal decision time for EXPECTED inputs
        elif decision_type == 'fit':
            timing_index = self.fit_decision_index[key,:]
        else:
            raise ValueError("decision_type must be \"optimal\" or \"fit\"")
        
        #! METRIC TYPE SHOULD BE TRUE ALMOST ALWAYS
        if metric_type == 'true':
            metric_type_index = 0
        elif metric_type == 'expected':
            metric_type_index = 1
        else:
            raise ValueError('metric_type must be \'true\' or \'expected\'')
        
        if metric2 is None: # For non-reaction/guess metrics
            ans = np.zeros(self.inputs.num_blocks)*np.nan
            for i in range(self.inputs.num_blocks):
                if metric1.ndim < 3:
                    ans[i] = metric1[i,timing_index[i]]
                else:
                    ans[i] = metric1[metric_type_index, i, timing_index[i]]
            return np.round(ans,self.inputs.round_num)
        else: # For reaction/guess perc metrics
            ans1 = np.zeros(self.inputs.num_blocks)*np.nan
            ans2 = np.zeros(self.inputs.num_blocks)*np.nan
            for i in range(self.inputs.num_blocks):
                if metric1.ndim < 3:
                    ans1[i] = metric1[i,timing_index[i]]
                    ans2[i] = metric2[i,timing_index[i]]
                else:
                    ans1[i] = metric1[metric_type_index, i, timing_index[i]]
                    ans2[i] = metric2[metric_type_index, i, timing_index[i]]
            return np.divide(np.round(ans1,self.inputs.round_num), np.round(ans2,self.inputs.round_num), 
                              out=np.zeros_like(ans2), where=ans2 > 1e-10)


class ModelConstructor:
    """
    Construct the model by sequentially creating each object, and passing it to the next object

    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.inputs = ModelInputs(**self.kwargs)

        self.agent_behavior   = AgentBehavior(self.inputs)
            
        self.player_behavior  = PlayerBehavior(self.inputs, self.agent_behavior)
        self.score_metrics    = ScoreMetrics(self.inputs, self.player_behavior, self.agent_behavior)
        self.results          = Results(self.inputs,self.score_metrics)
        
    def reset_model(self,skip_agent_behavior=False,**kwargs):
        self.kwargs = kwargs
        self.inputs.reset(**self.kwargs)
        if not skip_agent_behavior:
            self.agent_behavior.reset(self.inputs)
        self.player_behavior.reset(self.inputs,self.agent_behavior)
        self.score_metrics.reset(self.inputs,self.player_behavior,self.agent_behavior)
        self.results.reset(self.inputs,self.score_metrics)
        
    def fit_model(self, metric, target) -> None:
        '''
        Fitting the model with no free parameters
        - Just pick the decision time that minimizes the difference between model predicted movement onset
        and the data movement onset
        '''
        loss = abs(metric - target[np.newaxis,:, np.newaxis]) # Get the vectorized loss between metric (2,6,1800) and target (6,)
        decision_index = np.argmin(loss, axis=2)  # Across timesteps (axis=2) return the indices that minimize difference between metric and target
        self.results.set_fit_decision_index(decision_index.astype(int)) # Set the decision_index
        
        
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
        self.guess_leave_time_sd_store         = None
        
    def run_model_fit_procedure(self, free_params_init: dict, metric_keys: list, targets: np.ndarray,
                                ftol ,xtol, maxiter=None,maxfev=None, method='Nelder-Mead', bnds=None, 
                                drop_condition_from_loss=None,):
        self.loss_store = []
        self.optimal_decision_time_store = [] 
        self.leave_time_store = []
        self.leave_time_sd_store = []
        self.guess_leave_time_sd_store = []
        self.initial_guess = np.array(list(free_params_init.values())) # Get the free param values from dict and make an array, scipy will flatten it if it's 2D
        self.drop_condition_from_loss = drop_condition_from_loss
        
        num_params = len(self.initial_guess)
        if bnds is None:
            bnds = tuple([[0,200]])*num_params
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
            out = optimize.basinhopping(self.get_loss, self.initial_guess,niter=maxiter,
                                 minimizer_kwargs={'method':'Powell',
                                                   'args':(metric_keys, targets, free_params_init.keys())},
                                 stepsize=0.05
                                 )
            final_param_dict = dict(zip(free_params_init.keys(),out.x))
        elif method == "dualannealing":
            out = optimize.dual_annealing(self.get_loss,bnds,
                                          args = (metric_keys, targets, free_params_init.keys()),
                                          maxiter=100,)
            final_param_dict = dict(zip(free_params_init.keys(),out.x))            

        else:
            out = optimize.minimize(
                self.get_loss, self.initial_guess, 
                args=(metric_keys, targets, free_params_init.keys()), 
                method=method, bounds=None, 
                options = {
                    "ftol": ftol,
                    "xtol":xtol,
                    "disp":False,
                    "maxiter":maxiter,
                    "maxfev":maxfev,
                },
            )
            final_param_dict = dict(zip(free_params_init.keys(),out.x))
        
        #* Update model one last time
        self.update_model(final_param_dict)  
        # self.parameter_arr               = np.array(self.parameter_arr)
        # self.optimal_decision_time_store = np.array(self.optimal_decision_time_store)
        # self.leave_time_store            = np.array(self.leave_time_store)
        # self.leave_time_sd_store         = np.array(self.leave_time_sd_store)
        # self.guess_leave_time_sd_store   = np.array(self.guess_leave_time_sd_store)
        
        # ans = out.x + np.min(self.inputs.timesteps)
        # ans = out.x#.reshape(self.initial_param_shape)
        return out
    
    def get_loss(self, free_params_values, 
                 metric_keys, targets, 
                 free_params_keys, decision_type='optimal',
                 update=True):
        # Create dictionary back
        new_parameters_dict = dict(zip(free_params_keys,free_params_values))
                
        for k,v in new_parameters_dict.items():
            #* return big loss if anything is less than 0
            if v<0:
                self.loss_store.append(1e3)
                return 1e3 + abs(v)
            #* Return big loss if true is less than expected
            # if k.endswith("_true"):
            #     expected_key = k.replace("_true","_expected")
            #     if v < new_parameters_dict[expected_key]:
            #         self.loss_store.append(1e3)
            #         return 1e3 + abs(v)
            # if k == "timing_sd_expected":
            #     if v>self.model.inputs.timing_sd[0,0,0]:
            #         return 1e3 + abs(v)
        if update:
            self.parameter_arr.append(free_params_values)
            # Get the new arrays from the optimized free parameter inputs
            self.update_model(new_parameters_dict) 
        # Get each metric from results at that specific decision time
        model_metrics = np.zeros_like(targets)
        for i in range(targets.shape[0]): 
            if 'leave_time' in metric_keys[i]:
                model_metric = getattr(self.model.player_behavior, metric_keys[i])
                # Find the metric at optimal decision time
                #! Metric type always being 'true' means that the metric array we're using is ALWAYS the 'true' array. 
                model_metrics[i,:] = self.model.results.get_metric(model_metric, 
                                                                   decision_type=decision_type, 
                                                                   metric_type="true")  
            elif 'decision_time' in metric_keys[i]:
                model_metric = getattr(self.model.results,metric_keys[i])
                model_metrics[i,:] = model_metric
            else:
                model_metric = getattr(self.model.score_metrics, metric_keys[i])
                model_metrics[i,:] = self.model.results.get_metric(model_metric, 
                                                                   decision_type=decision_type, 
                                                                   metric_type="true")  # Find the metric at optimal decision time
        loss = lf.ape_loss(model_metrics, targets,)
        try:
            self.loss_store.append(loss)
        except AttributeError:
            pass
         
        # self.optimal_decision_time_store.append(self.model.results.optimal_decision_time[self.model.inputs.key]) # index at key bc I want the decision time for either expected or true
        # self.leave_time_store.append(self.model.results.get_metric(self.model.player_behavior.wtd_leave_time,
        #                                                            decision_type=decision_type,metric_type='true'))
        # self.leave_time_sd_store.append(self.model.results.get_metric(self.model.player_behavior.wtd_leave_time_sd,
        #                                                               decision_type=decision_type,metric_type='true'))
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
            # Try is for if I'm fitting the switch delay (which is an array of 2, so we assign to 0, which is true)
            # Except is for if I'm fitting the rewards associated with the scoring 
            try:
                #! Always changing the TRUE value, the expected value should always be set as of 8/15/23
                #! 9/26/23 - Now want to fit both the true and expected values, at least for guess_switch_delay and guess_switch_sd
                if '_true' in k:
                    self.model.kwargs[k.removesuffix('_true')][0] = v  # the 0 is for the true element of the array
                elif '_expected' in k:
                    self.model.kwargs[k.removesuffix('_expected')][1] = v # The 1 is for the expected element of the array
                else: # Else, change both to be the exact same thing
                    self.model.kwargs[k][0] = v
                    self.model.kwargs[k][1] = v
            except TypeError:
                print("Type Error")
                self.model.kwargs[k] = v
        # if self.model.kwargs["guess_switch_delay"].squeeze()[1] == self.model.kwargs["guess_switch_sd"].squeeze()[1]:
        #     print([v for v in free_param_dict.values()])
        #* Pass new set of kwargs to the reset_model which runs constructor sequence through resets, instead of re-instantiating
        if any(key in free_param_dict.keys() for key in ["timing_sd","timing_sd_expected","timing_sd_true"]):
            self.model.reset_model(skip_agent_behavior=False,**self.model.kwargs)
        else:
            self.model.reset_model(skip_agent_behavior=True,**self.model.kwargs)
