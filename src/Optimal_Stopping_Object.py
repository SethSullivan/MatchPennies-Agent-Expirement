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
                for k in range(len(timesteps)): # Looping here bc numba_scipy version of sc.erfc can only take float, not an array
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
            
            self.gamble_delay_known   = kwargs.get('gamble_delay_known',True)
            self.gamble_sd_known      = kwargs.get('gamble_sd_known')
            
            # Uncertainty
            self.reaction_sd          = kwargs.get('reaction_sd')
            self.movement_sd          = kwargs.get('movement_sd')
            self.timing_sd            = kwargs.get('timing_sd')
            self.gamble_decision_sd   = kwargs.get('gamble_decision_sd',{'true':np.array([50]*6),'exp':np.array([10]*6)})
            
            self.reaction_reach_sd = combine_sd_dicts(self.reaction_sd,self.movement_sd)
            self.gamble_reach_sd   = combine_sd_dicts(self.gamble_decision_sd,self.movement_sd)
            # Ability
            self.reaction_time               = kwargs.get('reaction_time')
            self.gamble_delay                = kwargs.get('gamble_delay',{'true':np.array([150]*6),'exp':np.array([50]*6)})
            self.movement_time               = kwargs.get('movement_time')
            self.reaction_plus_movement_time = add_dicts(self.reaction_time,self.movement_time)
            self.gamble_reach_time           = add_dicts(self.timesteps_dict,self.movement_time,self.gamble_delay)
            
            # Reward and cost values
            self.reward_matrix   = kwargs.get('reward_matrix',np.array([[1,0,0],[1,-1,0],[1,0,-1],[1,-1,-1]]))
            self.condition_one   = np.tile(self.reward_matrix[0],(1800,1))
            self.condition_two   = np.tile(self.reward_matrix[1],(1800,1))
            self.condition_three = np.tile(self.reward_matrix[2],(1800,1))
            self.condition_four  = np.tile(self.reward_matrix[3],(1800,1))
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
    def __init__(self, model_inputs: ModelInputs, er: ExpectedReward)->None:
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
    def __init__(self,**kwargs):
        model_inputs    = ModelInputs(**kwargs)
        agent_behavior  = AgentBehavior(model_inputs)
        player_behavior = PlayerBehavior(model_inputs,agent_behavior)
        
        score_metrics   = ScoreMetrics(model_inputs,player_behavior)
        expected_reward = ExpectedReward(model_inputs,score_metrics)
        optimal_output  = OptimalExpectedReward(model_inputs,expected_reward)
        calculator      = OptimalMetricsCalculator(optimal_output)

def main():        
    m = ModelConstructor(experiment='Exp1', num_blocks = 6, BETA_ON = False, numba = True,
                               agent_means = np.array([1000,1000,1100,1100,1200,1200]).astype(float),agent_sds = np.array([100]*6).astype(float), 
                               reaction_time = {'true':275,'exp':275}, movement_time = {'true':150,'exp':150},
                               reaction_sd = {'true':25,'exp':25}, movement_sd = {'true':25,'exp':25},
                               timing_sd = {'true':np.array([150]*6),'exp':np.array([150]*6)},
                               perc_wins_when_both_reach = np.array([0.8]*6),
                               gamble_delay_known = True, gamble_sd_known = True,
                               gamble_sd= {'true':150,'exp':10}, gamble_delay = {'true':125,'exp':50},
                                )
    # model_inputs = ModelInputs(experiment='Exp1', num_blocks = 6, BETA_ON = False, numba = True,
    #                            agent_means = np.array([1000,1000,1100,1100,1200,1200]).astype(float),agent_sds = np.array([100]*6).astype(float), 
    #                            reaction_time = {'true':275,'exp':275}, movement_time = {'true':150,'exp':150},
    #                            reaction_sd = {'true':25,'exp':25}, movement_sd = {'true':25,'exp':25},
    #                            timing_sd = {'true':np.array([150]*6),'exp':np.array([150]*6)},
    #                            perc_wins_when_both_reach = np.array([0.8]*6),
    #                            gamble_delay_known = True, gamble_sd_known = True,
    #                            gamble_sd= {'true':150,'exp':10}, gamble_delay = {'true':125,'exp':50},
    #                             )
    
    # agent_behavior  = AgentBehavior(model_inputs)
    # player_behavior = PlayerBehavior(model_inputs,agent_behavior)
    # score_metrics   = ScoreMetrics(model_inputs,player_behavior)
    # expected_reward = ExpectedReward(model_inputs,score_metrics)
    # optimal_output  = OptimalExpectedReward(model_inputs,expected_reward)
    # calculator      = OptimalMetricsCalculator(optimal_output)
    # print(calculator.find_optimal(score_metrics.prob_indecision))
    return 

if __name__ == '__main__':
    main()
    
#################################################################################################
#################################################################################################
###############################################################################################
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