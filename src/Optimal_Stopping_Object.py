import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.special as sc
import numba as nb
import numba_scipy # Needs to be imported so that numba recognizes scipy (specificall scipy special erf)
import data_visualization as dv
import copy
from numba_stats import norm
from functools import cached_property
wheel = dv.ColorWheel()
'''
04/04/23

v4 Takes away the Truncation stuff

04/17/23

Added in the flexibility to change reward around instead of agent mean and sd

'''

#####################################################
###### Helper Functions to get the Moments ##########
#####################################################
@nb.njit(parallel=True)
def get_moments(timesteps,agent_means,time_means,agent_sds,time_sds):
    EX_R,EX2_R,EX3_R = np.zeros((len(agent_means),len(time_means))),np.zeros((len(agent_means),len(time_means))),np.zeros((len(agent_means),len(time_means)))
    EX_G,EX2_G,EX3_G = np.zeros((len(agent_means),len(time_means))),np.zeros((len(agent_means),len(time_means))),np.zeros((len(agent_means),len(time_means)))
    for i in nb.prange(len(agent_means)):
        mu_x = agent_means[i]
        sig_x = agent_sds[i]
        for j in range(len(time_means)):
            sig_y = time_sds[i]
            mu_y = time_means[j]
            
            xpdf = (1/(sig_x*np.sqrt(2*np.pi)))*np.e**((-0.5)*((timesteps-mu_x)/sig_x)**2)
            prob_x_less_y = (sc.erfc((mu_x-mu_y)/(np.sqrt(2)*np.sqrt(sig_x**2 + sig_y**2))))/2
            prob_x_greater_y = 1 - prob_x_less_y # Or do the same as above and swap mu_x and mu_y
            if prob_x_less_y == 0:
                pass
            else:
                y_integrated = np.empty(len(timesteps),dtype=np.float64)
                y_inverse_integrated = np.empty(len(timesteps),dtype=np.float64)
                for k in range(len(timesteps)):
                    t = timesteps[k]
                    y_integrated[k] = (sc.erfc((t - mu_y)/(np.sqrt(2)*sig_y)))/2 # Going from x to infinity is the complementary error function (bc we want all the y's that are greater than x)
                    y_inverse_integrated[k] = (sc.erfc((mu_y - t)/(np.sqrt(2)*sig_y)))/2 # Swap limits of integration (mu_y - t) now
                EX_R[i,j] = np.sum(timesteps*xpdf*y_integrated)/prob_x_less_y
                EX2_R[i,j] = np.sum(timesteps**2*xpdf*y_integrated)/prob_x_less_y
                EX3_R[i,j] = np.sum(timesteps**3*xpdf*y_integrated)/prob_x_less_y
                
                EX_G[i,j] = np.sum(timesteps*xpdf*y_inverse_integrated)/prob_x_greater_y
                EX2_G[i,j] = np.sum(timesteps**2*xpdf*y_inverse_integrated)/prob_x_greater_y
                EX3_G[i,j] = np.sum(timesteps**3*xpdf*y_inverse_integrated)/prob_x_greater_y
        
    return EX_R,EX2_R,EX3_R,EX_G,EX2_G,EX3_G

def get_skew(EX,EX2,EX3):
    EX[EX==np.inf] == 100000
    EX2[EX2==np.inf] == 100000
    EX3[EX3==np.inf] == 100000
    ans = (EX3 - 3*EX*(EX2 - EX**2) - EX**3)/((EX2 - EX**2)**(3/2))
    return ans

def get_variance(EX,EX2):
    EX[EX==np.inf] == 100000
    EX2[EX2==np.inf] == 100000
    ans = EX2 - EX**2
    ans[ans<0] = np.nan
    return ans

# @nb.njit(parallel=True)
# TODO FIGURE OUT HOW TO USE THE WRAPPER HERE, then move on and decide how to implement expected vs true
def numba_cdf(x,mu_arr,sig_arr):
    if x.ndim==2: # If x dim is 1, then we have the x as the (6,1800)
        assert x.shape[0] == mu_arr.shape[0]
        ans = np.zeros(x.shape)
        for i in range(len(mu_arr)):
            ans[i,:] = norm.cdf(x[i,:],mu_arr[i],sig_arr[i])
    else: # Else, we have the mu_arr as the (6,1800) but we pass it in flattened
        ans = np.zeros(mu_arr.shape)
        for i in range(len(mu_arr)):
            ans[i] = norm.cdf(x,mu_arr[i],sig_arr[i])
    return ans
            
def add_dicts(d1,d2,d3=None):
    '''
    Takes the keys from d1 and uses that in the new dictionary.
    Then adds the values from d1 and d2
    '''
    if d3 is None:
        return {x1[0]:x1[1]+x2[1] for x1,x2 in zip(d1.items(),d2.items())}
    else:
        return {x1[0]:x1[1]+x2[1]+x3[1] for x1,x2,x3 in zip(d1.items(),d2.items(),d3.items())}
    
def combine_sd_dicts(d1,d2):
    d1_sq = {k:v**2 for k,v in d1.items()}
    d2_sq = {k:v**2 for k,v in d2.items()}
    temp = add_dicts(d1_sq,d2_sq)
    return {k:np.sqrt(v) for k,v in temp.items()}

def tile(arr,num):
    return np.tile(arr,(num,1)).T

class ModelInputs():
    def __init__(self, **kwargs):
        '''
        Model Inputs
        '''
        #* Task Conditions
        if True:
            self.experiment = kwargs.get('experiment')
            self.num_blocks  = kwargs.get('num_blocks')
            self.agent_means = kwargs.get('agent_means') # If exp2, need to be np.array([1100]*4)
            self.agent_sds   = kwargs.get('agent_sds') # If exp2, need to be np.array([50]*4)
            self.nsteps      = 1
            self.timesteps   = kwargs.get('timesteps',np.tile(np.arange(0.0,1800.0,self.nsteps),(self.num_blocks,1)))
            self.timesteps_dict = {'true':self.timesteps,'exp':self.timesteps}
            self.tiled_1500  = np.full_like(self.timesteps,1500.0)
            self.tiled_agent_means = np.tile(self.agent_means,(self.timesteps.shape[-1],1)).T
            self.tiled_agent_sds = np.tile(self.agent_sds,(self.timesteps.shape[-1],1)).T
            
            self.neg_inf_cut_off_value = -100000
            check = np.tile(np.arange(900.0,1100.0,self.nsteps),(self.num_blocks,1))
            # assert np.isclose(numba_cdf(check,np.array([5]*self.num_blocks), np.array([2]*self.num_blocks)),
            #                   stats.norm.cdf(check,np.array([5]),np.array([2]))).all()
            
        #* Player Parameters and Rewards
        if True:
            #  HOW MUCH PEOPLE WEIGH WINS VERSUS CORRECTNESS IS THE BETA TERM
            self.prob_win_when_both_reach  = kwargs.get('perc_wins_when_both_reach')/100
            # self.BETA_ON                   = kwargs.get('BETA_ON')
            # self.BETA = self.find_beta_term()
            
            self.gamble_delay_known               = kwargs.get('gamble_delay_known',True)
            self.gamble_sd_known         = kwargs.get('gamble_sd_known')
            
            # Uncertainty
            self.reaction_sd               = kwargs.get('reaction_sd')
            self.movement_sd               = kwargs.get('movement_sd')
            self.timing_sd                 = kwargs.get('timing_sd')
            self.gamble_decision_sd        = kwargs.get('gamble_decision_sd',{'true':np.array([50]*6),'exp':np.array([10]*6)})
            
            self.reaction_reach_sd = combine_sd_dicts(self.reaction_sd,self.movement_sd)
            self.gamble_reach_sd   = combine_sd_dicts(self.gamble_decision_sd,self.movement_sd)
            # Ability
            self.reaction_time               = kwargs.get('reaction_time')
            self.gamble_delay                = kwargs.get('gamble_delay',{'true':np.array([150]*6),'exp':np.array([50]*6)})
            self.movement_time               = kwargs.get('movement_time')
            self.reaction_plus_movement_time         = add_dicts(self.reaction_time,self.movement_time)
            self.gamble_reach_time           = add_dicts(self.timesteps_dict,self.movement_time,self.gamble_delay)
            
            # Reward and cost values
            self.reward_matrix = kwargs.get('reward_matrix',np.array([[1,0,0],[1,-1,0],[1,0,-1],[1,-1,-1]]))
            self.condition_one = np.tile(self.reward_matrix[0],(1800,1))
            self.condition_two = np.tile(self.reward_matrix[1],(1800,1))
            self.condition_three = np.tile(self.reward_matrix[2],(1800,1))
            self.condition_four = np.tile(self.reward_matrix[3],(1800,1))
            if self.experiment == 'Exp2':
                self.win_reward       = np.vstack((self.condition_one[:,0],self.condition_two[:,0],
                                                   self.condition_three[:,0],self.condition_four[:,0]))
                self.incorrect_cost   = np.vstack((self.condition_one[:,1],self.condition_two[:,1],
                                                   self.condition_three[:,1],self.condition_four[:,1]))
                self.indecision_cost  = np.vstack((self.condition_one[:,2],self.condition_two[:,2],
                                                   self.condition_three[:,2],self.condition_four[:,2]))
            else:
                self.win_reward      = kwargs.get('win_reward',1)
                self.incorrect_cost  = kwargs.get('incorrect_cost',0)
                self.indecision_cost = kwargs.get('indecision_cost',0)
            # Prob of selecting the correct target
            self.prob_selecting_correct_target_reaction = kwargs.get('prob_selecting_correct_target_reaction',1.0)
            self.prob_selecting_correct_target_gamble   = kwargs.get('prob_selecting_correct_target_gamble',0.5)

class AgentBehavior():
    def __init__(self, model_inputs:ModelInputs):
        self.inputs = model_inputs   
        self.cutoff_reaction_var        = None
        self.cutoff_reaction_skew       = None
        self.cutoff_agent_reaction_mean = None
        self.cutoff_agent_reaction_sd   = None
        
        self.cutoff_gamble_var          = None
        self.cutoff_gamble_skew         = None
        self.cutoff_agent_gamble_mean   = None
        self.cutoff_agent_gamble_sd     = None
        
        #* Get agent behavior
        self.cutoff_agent_behavior()
        
    @cached_property
    def prob_agent_has_gone(self):
        return numba_cdf(self.inputs.timesteps,self.inputs.agent_means,self.inputs.agent_sds)
    
    @cached_property
    def agent_moments(self):
        # Get first three central moments (EX2 is normalized for mean, EX3 is normalized for mean and sd) of the new distribution based on timing uncertainty
        inf_timesteps = np.arange(0,5000,1,dtype=np.float64)
        time_means = self.inputs.timesteps[0,:]
        return get_moments(inf_timesteps, self.inputs.agent_means, time_means,
                           self.inputs.agent_sds, self.inputs.timing_sd['exp'])

    def cutoff_agent_behavior(self):
        # Get the First Three moments for the left and right distributions (if X<Y and if X>Y respectively)
        EX_R,EX2_R,EX3_R,EX_G,EX2_G,EX3_G = self.agent_moments
        
        # Calculate the mean, variance, and skew with method of moments
        self.cutoff_agent_reaction_mean,self.cutoff_reaction_var,self.cutoff_reaction_skew = EX_R, get_variance(EX_R,EX2_R), get_skew(EX_R,EX2_R,EX3_R)
        self.cutoff_agent_reaction_sd = np.sqrt(self.cutoff_reaction_var)
        # same thing for gamble (X>Y)
        self.cutoff_agent_gamble_mean,self.cutoff_gamble_var,self.cutoff_gamble_skew = EX_G, get_variance(EX_G,EX2_G), get_skew(EX_G,EX2_G,EX3_G)
        self.cutoff_agent_gamble_sd = np.sqrt(self.cutoff_gamble_var)
    
class PlayerBehavior():
    '''
    This class contains the following for EVERY timestep
    
    1. Reaction/gamble leave/reach times and uncertainties
    2. Prob Selecting Reaction/Gamble
    3. Prob Making Given Reaction/Gamble
    '''
    def __init__(self,model_inputs: ModelInputs,agent_behavior: AgentBehavior,expected=True) -> None:
        self.inputs = model_inputs
        self.agent_behavior = agent_behavior
        
        if expected:
            self.key = 'exp'
        else:
            self.key = 'true'
        
        #* Leave times
        self.reaction_leave_time      = self.agent_behavior.cutoff_agent_reaction_mean + self.inputs.reaction_time[self.key]
        self.gamble_leave_time        = self.inputs.timesteps + self.inputs.gamble_delay[self.key]
        self.wtd_leave_target_time    = self.prob_selecting_reaction*self.reaction_leave_time + self.prob_selecting_gamble*self.gamble_leave_time
        #* Reach Times
        self.reaction_reach_time      = self.agent_behavior.cutoff_agent_reaction_mean + self.inputs.reaction_plus_movement_time[self.key]
        self.gamble_reach_time        = self.inputs.timesteps + self.inputs.gamble_delay[self.key] + self.inputs.movement_time[self.key]
        self.wtd_reach_target_time    = self.prob_selecting_reaction*self.reaction_reach_time + self.prob_selecting_gamble*self.gamble_reach_time
        
        #* Leave Time SD
        self.reaction_leave_time_sd   = np.sqrt(self.agent_behavior.cutoff_agent_reaction_sd**2 + self.inputs.reaction_sd[self.key]**2)
        self.gamble_leave_time_sd     = self.inputs.gamble_decision_sd[self.key]
        self.wtd_leave_target_time_sd = self.prob_selecting_reaction*self.reaction_leave_time_sd + self.prob_selecting_gamble*tile(self.gamble_leave_time_sd,self.inputs.timesteps.shape[-1])
        #* Reach Time SD
        self.reaction_reach_time_sd   = np.sqrt(self.reaction_leave_time_sd**2 + self.inputs.movement_sd[self.key]**2)
        self.gamble_reach_time_sd     = np.sqrt(self.gamble_leave_time_sd**2 + self.inputs.movement_sd[self.key]**2)
        self.wtd_reach_target_time_sd = self.prob_selecting_reaction*self.reaction_reach_time_sd + self.prob_selecting_gamble*tile(self.gamble_reach_time_sd,self.inputs.timesteps.shape[-1])

        #* Predict Decision Time
        self.predicted_decision_time = self.prob_selecting_reaction*self.agent_behavior.cutoff_agent_reaction_mean + \
                                        self.prob_selecting_gamble*self.agent_behavior.cutoff_agent_gamble_mean
        
    @cached_property
    def prob_selecting_reaction(self):
        combined_sd = np.sqrt(self.inputs.timing_sd[self.key]**2 + self.inputs.agent_sds**2) # Prob of SELECTING only includes timing uncertainty and agent uncertainty
        tiled_combined_sd = np.tile(combined_sd,(self.inputs.timesteps.shape[-1],1)).T
        diff = self.inputs.timesteps - self.inputs.tiled_agent_means
        ans = 1 - numba_cdf(np.array([0.0]),diff.flatten(),tiled_combined_sd.flatten()).reshape(self.inputs.timesteps.shape)
        return ans
    
    @cached_property
    def prob_selecting_gamble(self):
        return 1 - self.prob_selecting_reaction
    
    @cached_property 
    def prob_making_given_reaction(self):
        # Calculate the prob of making it on a reaction 
        #! Cutoff agent distribution isn't normal, so might not be able to simply add these, problem for later
        mu = self.reaction_reach_time
        sd = np.sqrt(self.agent_behavior.cutoff_agent_reaction_sd**2 + self.inputs.reaction_reach_sd[self.key]**2)
        temp = numba_cdf(np.array([1500]),mu.flatten(),sd.flatten())
        return temp.reshape(self.inputs.timesteps.shape)
    
    @cached_property
    def prob_making_given_gamble(self):
        mu = self.gamble_reach_time
        sd = np.tile(self.inputs.gamble_reach_sd[self.key],(self.inputs.timesteps.shape[-1],1)).T
        temp = numba_cdf(np.array([1500]), mu.flatten(),sd.flatten())
        return temp.reshape(self.inputs.timesteps.shape)
    
class ScoreMetrics():
    def __init__(self,model_inputs: ModelInputs,player_behavior: PlayerBehavior):
        self.inputs = model_inputs 
        #* These don't consider the probability that you select reaction
        # Prob of win
        self.prob_win_given_reaction        = self.inputs.prob_selecting_correct_target_reaction*player_behavior.prob_making_given_reaction
        self.prob_win_given_gamble          = self.inputs.prob_selecting_correct_target_gamble*player_behavior.prob_making_given_gamble
        
        # Prob of incorrect
        self.prob_incorrect_given_reaction  = (1 - self.inputs.prob_selecting_correct_target_reaction)*player_behavior.prob_making_given_reaction
        self.prob_incorrect_given_gamble    = (1 - self.inputs.prob_selecting_correct_target_gamble)*player_behavior.prob_making_given_gamble
        
        # Prob of indecision
        self.prob_indecision_given_reaction = 1 - player_behavior.prob_making_given_reaction
        self.prob_indecision_given_gamble   = 1 - player_behavior.prob_making_given_gamble
        
        # * Prob making on reaction and gamble depends on the prob of selecting reaction and gamble too
        self.prob_making_reaction      = player_behavior.prob_making_given_reaction*player_behavior.prob_selecting_reaction
        self.prob_making_gamble        = player_behavior.prob_making_given_gamble*player_behavior.prob_selecting_gamble
        self.prob_making               = self.prob_making_gamble + self.prob_making_reaction
        
        #* Multiply the actual probability of making it times the prob of getting it right for reaction and gamble
        self.prob_win_reaction         = self.prob_win_given_reaction*player_behavior.prob_selecting_reaction
        self.prob_win_gamble           = self.prob_win_given_gamble*player_behavior.prob_selecting_gamble
        self.prob_win                  = self.prob_win_reaction + self.prob_win_gamble
        
        #* Probability of receiving an incorrect cost
        self.prob_incorrect_reaction   = self.prob_incorrect_given_reaction*player_behavior.prob_selecting_reaction
        self.prob_incorrect_gamble     = self.prob_incorrect_given_gamble*player_behavior.prob_selecting_gamble
        self.prob_incorrect            = self.prob_incorrect_reaction + self.prob_incorrect_gamble
        
        #* Probability of receiving an indecision cost (No chance of success for indecision, so we just multiply by the two marginal probs calculated in last section)
        self.prob_indecision_reaction  = self.prob_indecision_given_reaction*player_behavior.prob_selecting_reaction
        self.prob_indecision_gamble    = self.prob_indecision_given_gamble*player_behavior.prob_selecting_gamble
        self.prob_indecision           =  self.prob_indecision_reaction + self.prob_indecision_gamble
        
        self.correct_decisions = player_behavior.prob_selecting_reaction*self.inputs.prob_selecting_correct_target_reaction +\
                                    player_behavior.prob_selecting_gamble*self.inputs.prob_selecting_correct_target_gamble 
                                    
        assert np.allclose(self.prob_win + self.prob_incorrect + self.prob_indecision, 1.0)

class ExpectedReward():
    def __init__(self,model_inputs,score_metrics: ScoreMetrics):
        self.inputs = model_inputs
        
        self.exp_reward_reaction    = score_metrics.prob_win_reaction*self.inputs.win_reward + \
                                        score_metrics.prob_incorrect_reaction*self.inputs.incorrect_cost + \
                                            score_metrics.prob_indecision_reaction*self.inputs.indecision_cost
        
        self.exp_reward_gamble     = score_metrics.prob_win_gamble*self.inputs.win_reward + \
                                        score_metrics.prob_incorrect_gamble*self.inputs.incorrect_cost + \
                                            score_metrics.prob_indecision_gamble*self.inputs.indecision_cost
                                            
        self.exp_reward            = score_metrics.prob_win*self.inputs.win_reward +\
                                        score_metrics.prob_incorrect*self.inputs.incorrect_cost + \
                                            score_metrics.prob_indecision*self.inputs.indecision_cost 

class OptimalExpectedReward():
    def __init__(self, model_inputs: ModelInputs, er: ExpectedReward):
        self.inputs = model_inputs
        # Find timepoint that gets the maximum expected reward
        self.optimal_index         = np.nanargmax(er.exp_reward,axis=1)
        self.optimal_decision_time = np.nanargmax(er.exp_reward, axis = 1)*self.inputs.nsteps
        self.max_exp_reward        = np.nanmax(er.exp_reward,axis=1)
        
        self.metrics_name_dict = {'exp_reward': 'Expected Reward','exp_reward_gamble': 'Expected Reward Gamble','exp_reward_reaction':'Expected Reward Reaction',
                                  'prob_making_reaction': 'Prob Making Reaction','prob_making_gamble':'Prob Making Gamble','prob_agent_has_gone':'Prob Agent Has Gone',
                                  'prob_selecting_reaction':'Prob of Selecting Reaction','prob_selecting_gamble':'Prob of Selecting Gamble',
                                  'prob_win_reaction':'Prob Win Reaction','prob_win_gamble':'Prob Win Gamble',
                                  'prob_incorrect_reaction':'Prob Incorrect Reaction','prob_incorrect_gamble':'Prob Incorrect Gamble',
                                  'prob_indecision_reaction':'Prob Indecision Reaction','prob_indecision_gamble': 'Prob Indecision Gamble',
                                  'prob_win':'Prob Win','prob_incorrect':'Prob Incorrect','prob_indecision':'Prob Indecision',
                                  'prob_making_reaction_based_on_agent':'Prob Making Based on Agent'}
    
class OptimalMetricsCalculator():
    '''
    This class contains 
    1. Find optimal function that uses the optimal index on the metrics calculated at every time step (From ScoreMetrics)
    2. Gets gamble/reaction calculations w/ first input being the gamble or reaction and second being the value that divides it
        - So we can get perc_reaction_wins which is (prob_win_reaction/prob_win)*100
    '''
    def __init__(self,optimal_output):
        self.optimal_index = optimal_output.optimal_decision_time
        
    def find_optimal(self,metric):
        ans = np.zeros(metric.shape[0])*np.nan
        for i in range(metric.shape[0]):
            ans[i] = metric[i,self.optimal_index[i]]
        return ans
    
    def gamble_reaction_metric(self,metric1,metric2):
        '''
        First metric is prob of that happening out of the second metric.
        np.divide handles the case where the denominator is 0 by just returning 0
        
        Example:
        Metric 1 = Prob Win Gamble 
        Metric 2 = Prob Win
        Out      = Perc Wins That Were Gamble (Out of all the wins, how many were gambles)
        '''

        arr1 = self.find_optimal(metric1)
        arr2 = self.find_optimal(metric2)
        return np.divide(arr1,arr2,out=np.zeros_like(arr2),where=arr2!=0)*100
 
class ModelConstructor():
    def __init__(self):
        model_inputs    = ModelInputs()
        agent_behavior  = AgentBehavior(model_inputs)
        player_behavior = PlayerBehavior(model_inputs,agent_behavior)
        
        score_metrics   = ScoreMetrics(model_inputs,player_behavior)
        expected_reward = ExpectedReward(model_inputs,score_metrics)
        optimal_output  = OptimalExpectedReward(model_inputs,expected_reward)

    
def main():        
   
    model_inputs = ModelInputs(experiment='Exp1', num_blocks = 6, BETA_ON = False, numba = True,
                               agent_means = np.array([1000,1000,1100,1100,1200,1200]).astype(float),agent_sds = np.array([100]*6).astype(float), 
                               reaction_time = {'true':275,'exp':275}, movement_time = {'true':150,'exp':150},
                               reaction_sd = {'true':25,'exp':25}, movement_sd = {'true':25,'exp':25},
                               timing_sd = {'true':np.array([150]*6),'exp':np.array([150]*6)},
                               perc_wins_when_both_reach = np.array([0.8]*6),
                               gamble_delay_known = True, gamble_sd_known = True,
                               gamble_sd= {'true':150,'exp':10}, gamble_delay = {'true':125,'exp':50},
                                )
    
    agent_behavior  = AgentBehavior(model_inputs)
    player_behavior = PlayerBehavior(model_inputs,agent_behavior)
    score_metrics   = ScoreMetrics(model_inputs,player_behavior)
    expected_reward = ExpectedReward(model_inputs,score_metrics)
    optimal_output  = OptimalExpectedReward(model_inputs,expected_reward)
    calculator      = OptimalMetricsCalculator(optimal_output)
    print(calculator.find_optimal(score_metrics.prob_indecision))
    return optimal_output

if __name__ == '__main__':
    main()
    
#################################################################################################
#################################################################################################
#################################################################################################

class Optimal_Decision_Time_Model():
    def __init__(self, **kwargs):
        '''
        Model Inputs
        '''
        # Task Conditions
        self.experiment = kwargs.get('experiment')
        self.num_blocks  = kwargs.get('num_blocks')
        self.agent_means = kwargs.get('agent_means',np.array([1000,1000,1100,1100,1200,1200])) # If exp2, need to be np.array([1100]*4)
        self.agent_sds   = kwargs.get('agent_sds',np.array([50,150,50,150,50,150])) # If exp2, need to be np.array([50]*4)
        self.nsteps      = 1
        self.timesteps   = kwargs.get('timesteps',np.tile(np.arange(0.0,1800.0,self.nsteps),(self.num_blocks,1)))
        
        self.tiled_1500  = np.full_like(self.timesteps,1500)
        self.tiled_agent_means = np.tile(self.agent_means,(self.timesteps.shape[-1],1)).T
        self.tiled_agent_sds = np.tile(self.agent_sds,(self.timesteps.shape[-1],1)).T
        
        self.neg_inf_cut_off_value = -100000
        check = np.tile(np.arange(900.0,1100.0,self.nsteps),(self.num_blocks,1))
        assert np.isclose(numba_cdf(check,np.array([5]*self.num_blocks),np.array([2]*self.num_blocks)),
                          stats.norm.cdf(check,np.array([5]),np.array([2]))).all()
        # * Model Variation Parameters:
        if True:
            self.true_gamble_delay                      = kwargs.get('true_gamble_delay')
            self.expected_gamble_delay                  = kwargs.get('expected_gamble_delay')
            self.gamble_delay_known                     = kwargs.get('gamble_delay_known')
            self.true_gamble_sd                = kwargs.get('true_gamble_sd')
            self.expected_gamble_sd            = kwargs.get('expected_gamble_sd')
            self.gamble_sd_known               = kwargs.get('gamble_sd_known')
            self.include_agent_sd_in_gamble_sd = kwargs.get('include_agent_sd_in_gamble_sd')
            self.weird_reaction_gamble_cutoff           = kwargs.get('weird_reaction_gamble_cutoff',0)
        #* Player Parameters and rewards
        if True:
            #  HOW MUCH PEOPLE WEIGH WINS VERSUS CORRECTNESS IS THE BETA TERM
            self.prob_win_when_both_reach  = kwargs.get('perc_wins_when_both_reach')/100
            self.BETA_ON                   = kwargs.get('BETA_ON')
            self.BETA = self.find_beta_term()

            # Uncertainty
            self.reaction_sd               = kwargs.get('reaction_sd')
            self.movement_sd               = kwargs.get('movement_sd')
            self.timing_sd                 = kwargs.get('timing_sd')
            self.reaction_reach_sd = np.sqrt(self.reaction_sd**2 + self.movement_sd**2)
            self.total_sd                  = np.sqrt(self.reaction_reach_sd**2 + self.timing_sd**2)
            self.total_sd_reaction         = self.reaction_reach_sd
            self.total_sd_gamble           = self.movement_sd 
            self.agent_plus_human_sd       = np.sqrt(self.total_sd**2 + self.agent_sds**2)
            # Ability
            self.reaction_time               = kwargs.get('reaction_time')
            self.movement_time               = kwargs.get('movement_time')
            self.reaction_plus_movement_time = self.reaction_time + self.movement_time
            # Reward and cost values
            self.reward_matrix = kwargs.get('reward_matrix',np.array([[1,0,0],[1,-1,0],[1,0,-1],[1,-1,-1]]))
            self.condition_one = np.tile(self.reward_matrix[0],(1800,1))
            self.condition_two = np.tile(self.reward_matrix[1],(1800,1))
            self.condition_three = np.tile(self.reward_matrix[2],(1800,1))
            self.condition_four = np.tile(self.reward_matrix[3],(1800,1))
            if self.experiment == 'Exp2':
                self.win_reward       = np.vstack((self.condition_one[:,0],self.condition_two[:,0],
                                                   self.condition_three[:,0],self.condition_four[:,0]))
                self.incorrect_cost   = np.vstack((self.condition_one[:,1],self.condition_two[:,1],
                                                   self.condition_three[:,1],self.condition_four[:,1]))
                self.indecision_cost  = np.vstack((self.condition_one[:,2],self.condition_two[:,2],
                                                   self.condition_three[:,2],self.condition_four[:,2]))
            else:
                self.win_reward      = kwargs.get('win_reward',1)
                self.incorrect_cost  = kwargs.get('incorrect_cost',0)
                self.indecision_cost = kwargs.get('indecision_cost',0)
            # Prob of selecting the correct target
            self.prob_selecting_correct_target_reaction = kwargs.get('prob_selecting_correct_target_reaction',1.0)
            self.prob_selecting_correct_target_gamble   = kwargs.get('prob_selecting_correct_target_gamble',0.5)
            
    #################################################################### 
    ################### ------Helper Functions-----#####################
    #################################################################### 
    def find_beta_term(self):
        '''
        Maps 0.5 (all gambles) to 0 and 1.0 to 1
        
        BETA = 2*wins_both_reached - 1
        
        This seems a bit volatile when implemented
        '''
        self.prob_win_when_both_reach = [0 if x<0.5 else x for x in self.prob_win_when_both_reach] # Anything under 0.5 becomes 0 
        return 2*np.array(self.prob_win_when_both_reach) - 1

    def replace_zero_with_nan(self,arr):
        arr[arr == 0] = np.nan
        return arr
    
    def plot_optimals(self,metrics,num_plots = None ,dpi=125):
        if num_plots is None:
            num_plots = self.num_blocks
        plt.style.use('cashaback_dark')
        for i in range(num_plots):
            fig,ax = plt.subplots(dpi=dpi)
            for metric in metrics:
                ax.plot(self.timesteps[i,:], getattr(self,metric)[i,:], label = metric,zorder=0)
                if metric == 'exp_reward':
                    ax.plot((self.optimal_decision_time[i],self.optimal_decision_time[i]),(-1,self.exp_reward[i,self.optimal_index[i]]),c='w')
                    ax.scatter(self.optimal_decision_time[i],self.exp_reward[i,self.optimal_index[i]],c='w')
                    ax.text(self.optimal_decision_time[i],self.exp_reward[i,self.optimal_index[i]]+0.03,f'Optimal Decision Time = {self.optimal_decision_time[i]}',ha = 'center')
            ax.set_ylim(np.min(self.incorrect_cost,self.indecision_cost)-0.03,np.max(self.win_reward)+0.03)
            ax.set_xlim(0,1500)
            ax.set_xticks(np.arange(0,1800,300))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Expected Reward')
            ax.legend(fontsize = 8,loc = (0.01,0.1))
            ax.set_title(f'Gain Function for Decision Time\nAgent Mean,SD = {self.agent_means[i]},{self.agent_sds[i]}')#\n B = {B}')
            plt.show()  
            
    #################################################################### 
    ################### ------ Run Model -----##########################
    ####################################################################
    def run_model(self):
        ###################### ----- Find Expected Reward Every Timestep -----  ######################
        #  Set true and expected gamble delay
        if self.gamble_sd_known:
            self.expected_gamble_sd = self.true_gamble_sd
        if self.gamble_delay_known:
            self.expected_gamble_delay = self.true_gamble_delay
            
        #* Get reward and probabilities at every timestep
        self.find_probabilities_based_on_agent_behavior()
        self.find_prob_win_incorrect_indecisions()
        self.find_expected_reward()
        
        ###################### ----- Get Experiment Metrics ----- #####################
        # * Calculate experiment metrics from expected reward 
        # (uses optimal index on all the calculations we already made to find the optimal)
        self.get_expected_experiment_metrics()
        
        # * Calculate the mean leave target time based on the decision time
        # Calculates reaction and gamble leave target times too
        self.calculate_mean_leave_target_time()
        
        # * Find the experiment metrics probabilistically 
        # (this is useful when calculating the wins people ACTUALLY get when there is something unknown
        # in the calculation of the gain function)
        self.get_true_experiment_metrics()
        
        self.calculate_true_and_expected_gamble_reaction_metrics()
        
    # * Probabilites ( Run through Functions)
    def find_probabilities_based_on_agent_behavior(self):
        #* Agent probabilities (not used, agent behavior is used in prob_of_selecting_reaction)
        self.prob_agent_has_gone            = self.prob_agent_go()
        self.prob_agent_has_not_gone        = 1 - self.prob_agent_has_gone
        
        #* Prob of selecting reacting or gambling decision
        self.prob_selecting_reaction        = self.prob_of_selecting_reaction() # Probability of SELECTING a Decision only depends on timing uncertainty, not total uncertainty
        self.prob_selecting_gamble          = 1 - self.prob_selecting_reaction
        
        # * Here we calculate the probability of making a reaction GIVEN we know that you selected reaction (conditional)
        # * THis uses the truncated agent distribution to determine if you'll make it or not
        # Prob of making it to the target
        self.prob_making_given_reaction     = self.prob_making_for_reaction()
        self.prob_making_given_gamble       = self.prob_making_for_gamble()
            
    def find_prob_win_incorrect_indecisions(self):        
        '''
        Prob of win, incorrect, indecisions (All equations, no functions)
        '''
        
        #* These don't consider the probability that you select reaction
        # Prob of win
        self.prob_win_given_reaction        = self.prob_selecting_correct_target_reaction*self.prob_making_given_reaction
        self.prob_win_given_gamble          = self.prob_selecting_correct_target_gamble*self.prob_making_given_gamble
        
        # Prob of incorrect
        self.prob_incorrect_given_reaction  = (1 - self.prob_selecting_correct_target_reaction)*self.prob_making_given_reaction
        self.prob_incorrect_given_gamble    = (1 - self.prob_selecting_correct_target_gamble)*self.prob_making_given_gamble
        
        # Prob of indecision
        self.prob_indecision_given_reaction = 1 - self.prob_making_given_reaction
        self.prob_indecision_given_gamble   = 1 - self.prob_making_given_gamble
        
        # * Prob making on reaction and gamble depends on the prob of selecting reaction and gamble too
        self.prob_making_reaction      = self.prob_making_given_reaction*self.prob_selecting_reaction
        self.prob_making_gamble        = self.prob_making_given_gamble*self.prob_selecting_gamble
        self.prob_making               = self.prob_making_gamble + self.prob_making_reaction
        
        #* Multiply the actual probability of making it times the prob of getting it right for reaction and gamble
        self.prob_win_reaction         = self.prob_win_given_reaction*self.prob_selecting_reaction
        self.prob_win_gamble           = self.prob_win_given_gamble*self.prob_selecting_gamble
        self.prob_win                  = self.prob_win_reaction + self.prob_win_gamble
        
        #* Probability of receiving an incorrect cost
        self.prob_incorrect_reaction   = self.prob_incorrect_given_reaction*self.prob_selecting_reaction
        self.prob_incorrect_gamble     = self.prob_incorrect_given_gamble*self.prob_selecting_gamble
        self.prob_incorrect            = self.prob_incorrect_reaction + self.prob_incorrect_gamble
        
        #* Probability of receiving an indecision cost (No chance of success for indecision, so we just multiply by the two marginal probs calculated in last section)
        self.prob_indecision_reaction  = self.prob_indecision_given_reaction*self.prob_selecting_reaction
        self.prob_indecision_gamble    = self.prob_indecision_given_gamble*self.prob_selecting_gamble
        self.prob_indecision           =  self.prob_indecision_reaction + self.prob_indecision_gamble
        
        assert np.allclose(self.prob_win + self.prob_incorrect + self.prob_indecision, 1.0)
    
    # * Expected reward calculation 
    def find_expected_reward(self):
        self.exp_reward_reaction    = (self.prob_win_reaction*self.win_reward + self.prob_incorrect_reaction*self.incorrect_cost + self.prob_indecision_reaction*self.indecision_cost)
        self.exp_reward_gamble     = (self.prob_win_gamble*self.win_reward + self.prob_incorrect_gamble*self.incorrect_cost + self.prob_indecision_gamble*self.indecision_cost)
        self.exp_reward            = self.prob_win*self.win_reward + self.prob_incorrect*self.incorrect_cost + self.prob_indecision*self.indecision_cost 
        
        # Find timepoint that gets the maximum expected reward
        self.optimal_index         = np.nanargmax(self.exp_reward,axis=1)
        self.optimal_decision_time = np.nanargmax(self.exp_reward, axis = 1)*self.nsteps
        self.max_exp_reward        = np.nanmax(self.exp_reward,axis=1)
        
        self.metrics_name_dict = {'exp_reward': 'Expected Reward','exp_reward_gamble': 'Expected Reward Gamble','exp_reward_reaction':'Expected Reward Reaction',
                                  'prob_making_reaction': 'Prob Making Reaction','prob_making_gamble':'Prob Making Gamble','prob_agent_has_gone':'Prob Agent Has Gone',
                                  'prob_selecting_reaction':'Prob of Selecting Reaction','prob_selecting_gamble':'Prob of Selecting Gamble',
                                  'prob_win_reaction':'Prob Win Reaction','prob_win_gamble':'Prob Win Gamble',
                                  'prob_incorrect_reaction':'Prob Incorrect Reaction','prob_incorrect_gamble':'Prob Incorrect Gamble',
                                  'prob_indecision_reaction':'Prob Indecision Reaction','prob_indecision_gamble': 'Prob Indecision Gamble',
                                  'prob_win':'Prob Win','prob_incorrect':'Prob Incorrect','prob_indecision':'Prob Indecision',
                                  'prob_making_reaction_based_on_agent':'Prob Making Based on Agent'}
    
    ######################################################################################################
    ########## ---- Functions to Calculate the Expected Reward at Every Timestep ----- ###################
    ######################################################################################################
    def prob_agent_go(self):
        '''
        Calculates the probability that the agent has gone by each timestep
        '''
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        output = numba_cdf(self.timesteps,self.agent_means,self.agent_sds)
        return output
    
    def prob_of_selecting_reaction(self): 
        '''
        This includes the timing uncertainty, and every calculation of win, indecision, incorrect is based 
        on this.
        
        Gamble uncertainty does not impact this.
        I think the gamble uncertainty is the uncertainty in the time it takes you 
        to go from decision to movement and when you switch it's really uncertain.
        '''

        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        combined_sd = np.sqrt(self.timing_sd**2 + self.agent_sds**2) # Prob of SELECTING only includes timing uncertainty and agent uncertainty
        tiled_combined_sd = np.tile(combined_sd,(self.timesteps.shape[-1],1)).T
        # I've determined that the decision time just needs to be after, doesn't necessarily need to be after some decision action delay
        diff = self.timesteps - self.tiled_agent_means
        temp = 1 - numba_cdf(np.array([0]),diff.flatten(),tiled_combined_sd.flatten())
        output = temp.reshape(self.timesteps.shape) 
        
        #* Keep this around if I need to make sure the numba version is correct
        # output_old = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        # for i in range(self.num_blocks):   
        #     diff = self.timesteps[i,:] - self.agent_means[i]
        #     diff = diff.astype(np.float64)
        #     cutoff = np.full_like(diff,self.weird_reaction_gamble_cutoff)
        #     sd = np.full_like(diff,combined_sd[i])
        #     output_old[i,:] = 1 - stats.norm.cdf(cutoff,diff,sd)    
        # assert np.isclose(output_old,output).all()
        
        return output
    
    def prob_making_for_reaction(self):
        # Get first three central moments (EX2 is normalized for mean, EX3 is normalized for mean and sd) of the new distribution based on timing uncertainty
        inf_timesteps = np.arange(0,5000,1,dtype=np.float64)
        time_means = self.timesteps[0,:]
        
        # Get the First Three moments for the left and right distributions (if X<Y and if X>Y respectively)
        EX_R,EX2_R,EX3_R,EX_G,EX2_G,self.EX3_G = get_moments(inf_timesteps,self.agent_means,time_means,self.agent_sds,self.timing_sd)
        
        # Calculate the mean, variance, and skew with method of moments
        self.cutoff_agent_reaction_mean,self.cutoff_var,self.cutoff_skew = EX_R,get_variance(EX_R,EX2_R),get_skew(EX_R,EX2_R,EX3_R)
        self.cutoff_agent_reaction_sd = np.sqrt(self.cutoff_var)
        
        # same thing for gamble (X>Y)
        self.cutoff_agent_gamble_mean,self.cutoff_var,self.cutoff_skew = EX_G,get_variance(EX_G,EX2_G),get_skew(EX_G,EX2_G,self.EX3_G)

        self.cutoff_agent_gamble_sd = np.sqrt(self.cutoff_var)
        
        # Calculate the prob of making it on a reaction 
        #! Cutoff agent distribution isn't normal, so might not be able to simply add these, problem for later
        mu = self.cutoff_agent_reaction_mean + self.reaction_plus_movement_time
        sd = np.sqrt(self.cutoff_agent_reaction_sd**2 + self.reaction_reach_sd**2)
        temp = numba_cdf(np.array([1500]),mu.flatten(),sd.flatten())
        prob_make_reaction = temp.reshape(self.timesteps.shape)    
        return prob_make_reaction
    
    def prob_making_for_gamble(self):
        self.gamble_reach_time_mean   = self.timesteps + self.movement_time + self.expected_gamble_delay
        if (self.expected_gamble_sd == self.true_gamble_sd).all():
            self.gamble_reach_time_sd = self.expected_gamble_sd
        else:
            self.gamble_reach_time_sd     = np.sqrt(self.expected_gamble_sd**2 + self.movement_sd**2 + self.timing_sd**2)        
        mu = self.gamble_reach_time_mean
        sd = np.tile(self.gamble_reach_time_sd,(self.timesteps.shape[-1],1)).T
        temp = numba_cdf(np.array([1500]),mu.flatten(),sd.flatten())
        output = temp.reshape(self.timesteps.shape)
        # assert np.isclose(output,output1).all(), ('Vector CDF NOT WORKING')
        return output
    
    ########################################################################
    ##########---- Functions to Calculate Optimal Metrics ------############
    ########################################################################
        # * Just get the probabilities from the ER calculations, 
    def get_expected_experiment_metrics(self):
        '''
        Get the index of the calculated probabilities
        
        This won't be correct if there are unknown uncertainties
        '''
        # if self.BETA_ON:
        #     self.ER_index = self.beta_optimal_index
        # else:
        #     self.ER_index = self.optimal_index
        list_attributes = [a for a in self.__dict__.keys() if (a.startswith('prob') or a.startswith('exp') or a.startswith('cutoff'))] # Get all attributes
        for a in list_attributes:
            new_attribute_value = np.zeros(self.num_blocks)
            new_attribute_name = 'optimal_expected_' + a 
            attribute = getattr(self,a)
            if isinstance(attribute,np.ndarray):
                if attribute.ndim == 2:
                    for i in range(self.num_blocks):
                        new_attribute_value[i] = attribute[i,self.optimal_index[i]]
                    setattr(self,new_attribute_name,new_attribute_value)
                    
    def calculate_mean_leave_target_time(self):
        # Optimal reaction leave target time is agent mean + reaction time
        self.optimal_expected_reaction_leave_target_time_mean = self.optimal_expected_cutoff_agent_reaction_mean + self.reaction_time
        self.optimal_expected_reaction_leave_target_time_sd   = np.sqrt(self.optimal_expected_cutoff_agent_reaction_sd**2 + self.reaction_sd**2)
        # Find optimal gamble leave target time and sd
        self.optimal_expected_gamble_leave_target_time_mean = self.optimal_decision_time + self.expected_gamble_delay
        if (self.expected_gamble_sd == self.true_gamble_sd).all():
            self.optimal_expected_gamble_leave_target_time_sd = self.expected_gamble_sd
        else:
            self.optimal_expected_gamble_leave_target_time_sd     = np.sqrt(self.expected_gamble_sd**2 + self.timing_sd**2)
        
        # Get the leave target time by weighing by how often they react and gamble
        self.wtd_optimal_expected_leave_target_time = (self.optimal_expected_prob_selecting_reaction*self.optimal_expected_reaction_leave_target_time_mean + \
                                            self.optimal_expected_prob_selecting_gamble*self.optimal_expected_gamble_leave_target_time_mean)/(self.optimal_expected_prob_selecting_gamble+self.optimal_expected_prob_selecting_reaction) 
        self.wtd_optimal_expected_sd_leave_target_time = np.sqrt(self.optimal_expected_prob_selecting_reaction*(self.optimal_expected_reaction_leave_target_time_sd**2)+\
                                                                    self.optimal_expected_prob_selecting_gamble*(self.optimal_expected_gamble_leave_target_time_sd**2))
        # Just add movement time for reach target time
        self.wtd_optimal_expected_reach_target_time = self.wtd_optimal_expected_leave_target_time + self.movement_time
        
        # Predict their decision time
        self.expected_predicted_decision_time = (self.optimal_expected_prob_selecting_reaction*self.optimal_expected_cutoff_agent_reaction_mean) + \
                                       (self.optimal_expected_prob_selecting_gamble*(self.optimal_expected_gamble_leave_target_time_mean - self.expected_gamble_delay))
                                       
        self.expected_player_minus_agent_leave_time          = self.wtd_optimal_expected_leave_target_time - self.agent_means
        self.expected_player_minus_agent_reaction_leave_time = self.optimal_expected_reaction_leave_target_time_mean - self.optimal_expected_cutoff_agent_reaction_mean
        self.expected_player_minus_agent_gamble_leave_time   = self.optimal_expected_gamble_leave_target_time_mean - self.optimal_expected_cutoff_agent_gamble_mean
        
        self.optimal_expected_phat_correct = self.optimal_expected_prob_selecting_reaction*1.0 + self.optimal_expected_prob_selecting_gamble*0.5

        ############################ TRUE ##########################################
        self.optimal_true_prob_selecting_reaction             = self.optimal_expected_prob_selecting_reaction
        self.optimal_true_prob_selecting_gamble               = self.optimal_expected_prob_selecting_gamble
        self.optimal_true_reaction_leave_target_time_mean     = self.optimal_expected_reaction_leave_target_time_mean
        self.optimal_true_reaction_leave_target_time_sd       = self.optimal_expected_reaction_leave_target_time_sd
        self.optimal_true_gamble_leave_target_time_mean       = self.optimal_decision_time + self.true_gamble_delay
        self.optimal_true_gamble_leave_target_time_sd         = np.sqrt(self.true_gamble_sd**2)
        
         # Get the leave target time by weighing by how often they react and gamble
        self.wtd_optimal_true_leave_target_time = (self.optimal_true_prob_selecting_reaction*self.optimal_true_reaction_leave_target_time_mean + \
                                            self.optimal_true_prob_selecting_gamble*self.optimal_true_gamble_leave_target_time_mean)/(self.optimal_true_prob_selecting_gamble+self.optimal_true_prob_selecting_reaction) 
        self.wtd_optimal_true_sd_leave_target_time = np.sqrt(self.optimal_true_prob_selecting_reaction*(self.optimal_true_reaction_leave_target_time_sd**2)+\
                                                                    self.optimal_true_prob_selecting_gamble*(self.optimal_true_gamble_leave_target_time_sd**2))
        # Just add movement time for reach target time
        self.wtd_optimal_true_reach_target_time = self.wtd_optimal_true_leave_target_time + self.movement_time
        
        # Predict their decision time
        self.true_predicted_decision_time = (self.optimal_true_prob_selecting_reaction*self.optimal_expected_cutoff_agent_reaction_mean) + \
                                       (self.optimal_true_prob_selecting_gamble*(self.optimal_true_gamble_leave_target_time_mean - self.true_gamble_delay))
                                       
        self.true_player_minus_agent_leave_time          = self.wtd_optimal_true_leave_target_time - self.agent_means
        self.true_player_minus_agent_reaction_leave_time = self.optimal_true_reaction_leave_target_time_mean - self.optimal_expected_cutoff_agent_reaction_mean # Agent cutoff doesn't change depending on gamble delay
        self.true_player_minus_agent_gamble_leave_time   = self.optimal_true_gamble_leave_target_time_mean - self.optimal_expected_cutoff_agent_gamble_mean
        
        self.optimal_true_phat_correct = self.optimal_true_prob_selecting_reaction*1.0 + self.optimal_true_prob_selecting_gamble*0.5

        return
    
    # * Calculate experiment metrics possibly with unknown gamble stuff
    def get_true_experiment_metrics(self):   
        def _optimal_true_prob_making_for_reaction_and_gamble():
            '''
            Takes into account (un)known gamble delay
            '''
            
            # Reaction calc
            self.optimal_true_reaction_plus_movement_time = self.optimal_expected_cutoff_agent_reaction_mean + self.reaction_plus_movement_time
            self.reaction_plus_movement_time_sd = np.sqrt(self.optimal_expected_cutoff_agent_reaction_sd**2 + self.reaction_reach_sd**2)
            output1 = numba_cdf(np.array([1500]),self.optimal_true_reaction_plus_movement_time,self.reaction_plus_movement_time_sd)
            
            # Gamble Calc
            self.optimal_true_gamble_reach_time_mean = self.optimal_decision_time + self.true_gamble_delay + self.movement_time
            # ! True Gamble uncertainty is from data, which includes the agent's uncertainty, so we don't add that here 
            self.optimal_true_gamble_reach_sd = np.sqrt(self.true_gamble_sd**2 + self.movement_sd**2) 
            output2 = numba_cdf(np.array([1500]),self.optimal_true_gamble_reach_time_mean,self.optimal_true_gamble_reach_sd)
            return output1,output2
         
        if True:
            # ! Here we calculate the probability of making a reaction GIVEN we know that you selected reaction (conditional)
            # Prob of making it to the target
            self.optimal_true_prob_making_given_reaction,self.optimal_true_prob_making_given_gamble = _optimal_true_prob_making_for_reaction_and_gamble()
            
            # Prob of win
            self.optimal_true_prob_win_given_reaction = self.prob_selecting_correct_target_reaction*self.optimal_true_prob_making_given_reaction
            self.optimal_true_prob_win_given_gamble = self.prob_selecting_correct_target_gamble*self.optimal_true_prob_making_given_gamble
            
            # Prob of indecision
            self.optimal_true_prob_indecision_given_reaction = 1 - self.optimal_true_prob_making_given_reaction
            self.optimal_true_prob_indecision_given_gamble = 1 - self.optimal_true_prob_making_given_gamble
            
            # Prob of incorrect
            self.optimal_true_prob_incorrect_given_reaction = (1 - self.prob_selecting_correct_target_reaction)*self.optimal_true_prob_making_given_reaction
            self.optimal_true_prob_incorrect_given_gamble = (1 - self.prob_selecting_correct_target_gamble)*self.optimal_true_prob_making_given_gamble
            
            # Doesn't matter if you make it, what's the probability you intend to go in the correct direction

        # * Prob of win, incorrect, indecisions (All equations, no functions)
        if True:
             #* These don't consider the probability that you select reaction
            # Prob of win
            self.optimal_true_prob_win_given_reaction        = self.prob_selecting_correct_target_reaction*self.optimal_true_prob_making_given_reaction
            self.optimal_true_prob_win_given_gamble          = self.prob_selecting_correct_target_gamble*self.optimal_true_prob_making_given_gamble
            
            # Prob of indecision
            self.optimal_true_prob_indecision_given_reaction = 1 - self.optimal_true_prob_making_given_reaction
            self.optimal_true_prob_indecision_given_gamble   = 1 - self.optimal_true_prob_making_given_gamble
            
            # Prob of incorrect
            self.optimal_true_prob_incorrect_given_reaction  = (1 - self.prob_selecting_correct_target_reaction)*self.optimal_true_prob_making_given_reaction
            self.optimal_true_prob_incorrect_given_gamble    = (1 - self.prob_selecting_correct_target_gamble)*self.optimal_true_prob_making_given_gamble
            
            # * Prob making on reaction and gamble depends on the prob of selecting reaction and gamble too
            self.optimal_true_prob_making_reaction      = self.optimal_true_prob_making_given_reaction*self.optimal_true_prob_selecting_reaction
            self.optimal_true_prob_making_gamble        = self.optimal_true_prob_making_given_gamble*self.optimal_true_prob_selecting_gamble
            self.optimal_true_prob_making               = self.optimal_true_prob_making_gamble + self.optimal_true_prob_making_reaction
            
            #* Multiply the actual probability of making it times the prob of getting it right for reaction and gamble
            self.optimal_true_prob_win_reaction         = self.prob_selecting_correct_target_reaction*self.optimal_true_prob_making_given_reaction*self.optimal_true_prob_selecting_reaction
            self.optimal_true_prob_win_gamble           = self.prob_selecting_correct_target_gamble*self.optimal_true_prob_making_given_gamble*self.optimal_true_prob_selecting_gamble
            self.optimal_true_prob_win                  = self.optimal_true_prob_win_reaction + self.optimal_true_prob_win_gamble
            
            #* Probability of receiving an incorrect cost
            self.optimal_true_prob_incorrect_reaction   = (1 - self.prob_selecting_correct_target_reaction)*self.optimal_true_prob_making_given_reaction*self.optimal_true_prob_selecting_reaction
            self.optimal_true_prob_incorrect_gamble     = (1 - self.prob_selecting_correct_target_gamble)*self.optimal_true_prob_making_given_gamble*self.optimal_true_prob_selecting_gamble
            self.optimal_true_prob_incorrect            = self.optimal_true_prob_incorrect_reaction + self.optimal_true_prob_incorrect_gamble
            
            #* Probability of receiving an indecision cost (No chance of success for indecision, so we just multiply by the two marginal probs calculated in last section)
            self.optimal_true_prob_indecision_reaction  = (1 - self.optimal_true_prob_making_given_reaction)*self.optimal_true_prob_selecting_reaction
            self.optimal_true_prob_indecision_gamble    = (1 - self.optimal_true_prob_making_given_gamble)*self.optimal_true_prob_selecting_gamble
            self.optimal_true_prob_indecision           = 1 - self.optimal_true_prob_making             

    def calculate_true_and_expected_gamble_reaction_metrics(self):
        if True:
            # Percent of metric that were reaction and gamble
            self.optimal_true_prob_wins_that_were_gamble           = np.divide(self.optimal_true_prob_win_gamble,self.optimal_true_prob_win,out=np.zeros_like(self.optimal_true_prob_win),where=self.optimal_true_prob_win!=0)
            self.optimal_true_prob_indecisions_that_were_gamble    = np.divide(self.optimal_true_prob_indecision_gamble,self.optimal_true_prob_indecision,out=np.zeros_like(self.optimal_true_prob_indecision),where=self.optimal_true_prob_indecision!=0)
            self.optimal_true_prob_incorrects_that_were_gamble     = np.divide(self.optimal_true_prob_incorrect_gamble,self.optimal_true_prob_incorrect,out=np.zeros_like(self.optimal_true_prob_incorrect),where=self.optimal_true_prob_incorrect!=0)
            
            self.optimal_true_prob_wins_that_were_reaction         = np.divide(self.optimal_true_prob_win_reaction,self.optimal_true_prob_win,out=np.zeros_like(self.optimal_true_prob_win),where=self.optimal_true_prob_win!=0)
            self.optimal_true_prob_indecisions_that_were_reaction  = np.divide(self.optimal_true_prob_indecision_reaction,self.optimal_true_prob_indecision,out=np.zeros_like(self.optimal_true_prob_indecision),where=self.optimal_true_prob_indecision!=0)
            self.optimal_true_prob_incorrects_that_were_reaction   = np.divide(self.optimal_true_prob_incorrect_reaction,self.optimal_true_prob_incorrect,out=np.zeros_like(self.optimal_true_prob_incorrect),where=self.optimal_true_prob_incorrect!=0)
            
            # Probent of reaction or gamble that were wins/incorrects/indecisions

            self.optimal_true_prob_gambles_that_were_wins          = np.divide(self.optimal_true_prob_win_gamble,self.optimal_expected_prob_selecting_gamble,out=np.zeros_like(self.optimal_expected_prob_selecting_gamble),where=self.optimal_expected_prob_selecting_gamble!=0)
            self.optimal_true_prob_gambles_that_were_incorrects    = np.divide(self.optimal_true_prob_incorrect_gamble,self.optimal_expected_prob_selecting_gamble,out=np.zeros_like(self.optimal_expected_prob_selecting_gamble),where=self.optimal_expected_prob_selecting_gamble!=0)
            self.optimal_true_prob_gambles_that_were_indecisions   = np.divide(self.optimal_true_prob_indecision_gamble,self.optimal_expected_prob_selecting_gamble,out=np.zeros_like(self.optimal_expected_prob_selecting_gamble),where=self.optimal_expected_prob_selecting_gamble!=0)
            
            self.optimal_true_prob_reactions_that_were_wins        = np.divide(self.optimal_true_prob_win_reaction,self.optimal_expected_prob_selecting_reaction,out=np.zeros_like(self.optimal_expected_prob_selecting_reaction),where=self.optimal_expected_prob_selecting_reaction!=0)
            self.optimal_true_prob_reactions_that_were_incorrects  = np.divide(self.optimal_true_prob_incorrect_reaction,self.optimal_expected_prob_selecting_reaction,out=np.zeros_like(self.optimal_expected_prob_selecting_reaction),where=self.optimal_expected_prob_selecting_reaction!=0)
            self.optimal_true_prob_reactions_that_were_indecisions = np.divide(self.optimal_true_prob_indecision_reaction,self.optimal_expected_prob_selecting_reaction,out=np.zeros_like(self.optimal_expected_prob_selecting_reaction),where=self.optimal_expected_prob_selecting_reaction!=0)
        
        if True:
            # Percent of metric that were reaction and gamble
            self.optimal_expected_prob_wins_that_were_gamble           = np.divide(self.optimal_expected_prob_win_gamble,self.optimal_expected_prob_win,out=np.zeros_like(self.optimal_expected_prob_win),where=self.optimal_expected_prob_win!=0)
            self.optimal_expected_prob_indecisions_that_were_gamble    = np.divide(self.optimal_expected_prob_indecision_gamble,self.optimal_expected_prob_indecision,out=np.zeros_like(self.optimal_expected_prob_indecision),where=self.optimal_expected_prob_indecision!=0)
            self.optimal_expected_prob_incorrects_that_were_gamble     = np.divide(self.optimal_expected_prob_incorrect_gamble,self.optimal_expected_prob_incorrect,out=np.zeros_like(self.optimal_expected_prob_incorrect),where=self.optimal_expected_prob_incorrect!=0)
            
            self.optimal_expected_prob_wins_that_were_reaction         = np.divide(self.optimal_expected_prob_win_reaction,self.optimal_expected_prob_win,out=np.zeros_like(self.optimal_expected_prob_win),where=self.optimal_expected_prob_win!=0)
            self.optimal_expected_prob_indecisions_that_were_reaction  = np.divide(self.optimal_expected_prob_indecision_reaction,self.optimal_expected_prob_indecision,out=np.zeros_like(self.optimal_expected_prob_indecision),where=self.optimal_expected_prob_indecision!=0)
            self.optimal_expected_prob_incorrects_that_were_reaction   = np.divide(self.optimal_expected_prob_incorrect_reaction,self.optimal_expected_prob_incorrect,out=np.zeros_like(self.optimal_expected_prob_incorrect),where=self.optimal_expected_prob_incorrect!=0)
            
            # Probent of reaction or gamble that were wins/incorrects/indecisions

            self.optimal_expected_prob_gambles_that_were_wins          = np.divide(self.optimal_expected_prob_win_gamble,self.optimal_expected_prob_selecting_gamble,out=np.zeros_like(self.optimal_expected_prob_selecting_gamble),where=self.optimal_expected_prob_selecting_gamble!=0)
            self.optimal_expected_prob_gambles_that_were_incorrects    = np.divide(self.optimal_expected_prob_incorrect_gamble,self.optimal_expected_prob_selecting_gamble,out=np.zeros_like(self.optimal_expected_prob_selecting_gamble),where=self.optimal_expected_prob_selecting_gamble!=0)
            self.optimal_expected_prob_gambles_that_were_indecisions   = np.divide(self.optimal_expected_prob_indecision_gamble,self.optimal_expected_prob_selecting_gamble,out=np.zeros_like(self.optimal_expected_prob_selecting_gamble),where=self.optimal_expected_prob_selecting_gamble!=0)
            
            self.optimal_expected_prob_reactions_that_were_wins        = np.divide(self.optimal_expected_prob_win_reaction,self.optimal_expected_prob_selecting_reaction,out=np.zeros_like(self.optimal_expected_prob_selecting_reaction),where=self.optimal_expected_prob_selecting_reaction!=0)
            self.optimal_expected_prob_reactions_that_were_incorrects  = np.divide(self.optimal_expected_prob_incorrect_reaction,self.optimal_expected_prob_selecting_reaction,out=np.zeros_like(self.optimal_expected_prob_selecting_reaction),where=self.optimal_expected_prob_selecting_reaction!=0)
            self.optimal_expected_prob_reactions_that_were_indecisions = np.divide(self.optimal_expected_prob_indecision_reaction,self.optimal_expected_prob_selecting_reaction,out=np.zeros_like(self.optimal_expected_prob_selecting_reaction),where=self.optimal_expected_prob_selecting_reaction!=0)
        
    ###########################################################
    #############----- Fit Model Functions ---- ###############
    ###########################################################
    def calculate_metrics_with_certain_decision_time(self,decision_times,final = False):
        '''
        Just recalculate new metrics with each decision time, keeping the expected reward function
        '''
        self.optimal_decision_time = decision_times
        self.expected_reward_of_decision_time = np.zeros((self.num_blocks))
        self.optimal_index = (decision_times/self.nsteps).astype(int)
        if final:
            for i in range(self.num_blocks):
                self.expected_reward_of_decision_time[i] = self.exp_reward[i,int(self.optimal_decision_time[i])]
        
        #* Recalculate
        self.get_expected_experiment_metrics()
        self.calculate_mean_leave_target_time()
        self.get_true_experiment_metrics()
        self.calculate_true_and_expected_gamble_reaction_metrics()
        
    def mse_loss_single(self,metric_name,target,decision_time,true=True):
        self.calculate_metrics_with_certain_decision_time(decision_time)
        model_data = getattr(self,metric_name)
        loss = abs(model_data - target)
        return loss
    
    def fit_model(self,metric_name,target):
        self.tune_timesteps = np.arange(900,1800,1)
        decision_times      = np.array([self.tune_timesteps[0]]*self.num_blocks) 
        loss_store          = np.zeros((self.num_blocks,len(self.tune_timesteps))) # Each metric,each block, each timestep

        for i in range(self.num_blocks):
            for j,t in enumerate(self.tune_timesteps):
                decision_times[i] = t
                loss_store[i,j] = self.mse_loss_single(metric_name,target,decision_times)[i]
        self.fit_decision_times = np.argmin(loss_store,axis=1) + np.min(self.tune_timesteps)
        self.calculate_metrics_with_certain_decision_time(self.fit_decision_times,final=True)  
   
@nb.njit(parallel=True)
def find_optimal_decision_time_for_certain_metric(ob,metric_name = 'RPMT'):
    '''
    Trying to search across the entire space of reaction and movement times and find the optimal decision time for that person 
    '''
    # o = copy.deepcopy(ob)
    rts = np.arange(220,400,1,np.int64)
    mts = np.arange(100,300,1,np.int64)
    ans = np.zeros((len(rts),len(mts),ob.num_blocks))

    for i in nb.prange(len(rts)):
        for j in nb.prange(len(mts)):
            ob.reaction_time = rts[i]
            ob.movement_time = mts[j]
            ob.run_model()
            ans[i,j,:] = ob.optimal_decision_time     