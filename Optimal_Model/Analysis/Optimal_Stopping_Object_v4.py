import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.special as sc
import numba_scipy
import numba as nb
import data_visualization as dv
from numba import njit
import copy
from numba_stats import norm
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
@njit(parallel=True)
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
    return EX2 - EX**2

class Optimal_Decision_Time_Model():
    def __init__(self, **kwargs):
        # Task Conditions
        self.experiment = kwargs.get('experiment')
        self.num_blocks  = kwargs.get('num_blocks')
        self.agent_means = kwargs.get('agent_means',np.array([1000,1000,1100,1100,1200,1200])) # If exp2, need to be np.array([1100]*4)
        self.agent_sds   = kwargs.get('agent_sds',np.array([50,150,50,150,50,150])) # If exp2, need to be np.array([50]*4)
        self.nsteps      = 1
        self.timesteps   = kwargs.get('timesteps',np.tile(np.arange(0,2000,self.nsteps),(self.num_blocks,1)))
        self.neg_inf_cut_off_value = -100000
        # * Model Variation Parameters:
        if True:
            self.gamble_delay_value                 = kwargs.get('gamble_delay_value')
            self.gamble_delay_known                 = kwargs.get('gamble_delay_known')
            self.gamble_uncertainty_value           = kwargs.get('gamble_uncertainty_value')
            self.gamble_uncertainty_known           = kwargs.get('gamble_uncertainty_known')
            self.decision_action_delay_mean         = kwargs.get('decision_action_delay_mean')
            self.decision_action_delay_uncertainty         = kwargs.get('decision_action_delay_uncertainty')
            self.weird_reaction_gamble_cutoff = kwargs.get('weird_reaction_gamble_cutoff',0)
        #* Player Parameters and rewards
        if True:
            #  HOW MUCH PEOPLE WEIGH WINS VERSUS CORRECTNESS IS THE BETA TERM
            self.prob_win_when_both_reach  = kwargs.get('perc_wins_when_both_reach')/100
            self.BETA_ON                   = kwargs.get('BETA_ON')
            self.BETA = self.find_beta_term()

            # Uncertainty
            self.reaction_uncertainty               = kwargs.get('reaction_uncertainty')
            self.movement_uncertainty               = kwargs.get('movement_uncertainty')
            self.timing_uncertainty                 = kwargs.get('timing_uncertainty')
            self.reaction_plus_movement_uncertainty = np.sqrt(self.reaction_uncertainty**2 + self.movement_uncertainty**2)
            self.total_uncertainty                  = np.sqrt(self.reaction_plus_movement_uncertainty**2 + self.timing_uncertainty**2)
            self.total_uncertainty_reaction         = self.reaction_plus_movement_uncertainty
            self.total_uncertainty_gamble           = self.movement_uncertainty 
            self.agent_plus_human_uncertainty       = np.sqrt(self.total_uncertainty**2 + self.agent_sds**2)
            # Ability
            self.reaction_time               = kwargs.get('reaction_time')
            self.movement_time               = kwargs.get('movement_time')
            self.reaction_plus_movement_time = self.reaction_time + self.movement_time
            # Reward and cost values
            self.reward_matrix = kwargs.get('reward_matrix',np.array([[1,0,0],[1,-1,0],[1,0,-1],[1,-1,-1]]))
            self.condition_one = np.tile(self.reward_matrix[0],(2000,1))
            self.condition_two = np.tile(self.reward_matrix[1],(2000,1))
            self.condition_three = np.tile(self.reward_matrix[2],(2000,1))
            self.condition_four = np.tile(self.reward_matrix[3],(2000,1))
            if self.experiment == 'Exp2':
                self.win_reward      = np.vstack((self.condition_one[:,0],self.condition_two[:,0],
                                                   self.condition_three[:,0],self.condition_four[:,0]))
                self.incorrect_cost      = np.vstack((self.condition_one[:,1],self.condition_two[:,1],
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
            ax.set_xticks(np.arange(0,2000,300))
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
        #* Set true and expected gamble delay
        self.true_gamble_delay = self.gamble_delay_value
        if self.gamble_delay_known:
            self.expected_gamble_delay = self.true_gamble_delay
        else:
            self.expected_gamble_delay = self.decision_action_delay_mean
        
        self.true_gamble_uncertainty = self.gamble_uncertainty_value
        if self.gamble_uncertainty_known:
            self.expected_gamble_uncertainty = self.true_gamble_uncertainty
        else:
            self.expected_gamble_uncertainty = np.sqrt(self.decision_action_delay_uncertainty**2 + self.agent_sds**2)
        
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
        
        # Prob of indecision
        self.prob_indecision_given_reaction = 1 - self.prob_making_given_reaction
        self.prob_indecision_given_gamble   = 1 - self.prob_making_given_gamble
        
        # Prob of incorrect
        self.prob_incorrect_given_reaction  = (1 - self.prob_selecting_correct_target_reaction)*self.prob_making_given_reaction
        self.prob_incorrect_given_gamble    = (1 - self.prob_selecting_correct_target_gamble)*self.prob_making_given_gamble
        
        # * Prob making on reaction and gamble depends on the prob of selecting reaction and gamble too
        self.prob_making_reaction      = self.prob_making_given_reaction*self.prob_selecting_reaction
        self.prob_making_gamble        = self.prob_making_given_gamble*self.prob_selecting_gamble
        self.prob_making               = self.prob_making_gamble + self.prob_making_reaction
        
        #* Multiply the actual probability of making it times the prob of getting it right for reaction and gamble
        self.prob_win_reaction         = self.prob_selecting_correct_target_reaction*self.prob_making_given_reaction*self.prob_selecting_reaction
        self.prob_win_gamble           = self.prob_selecting_correct_target_gamble*self.prob_making_given_gamble*self.prob_selecting_gamble
        self.prob_win                  = self.prob_win_reaction + self.prob_win_gamble
        
        #* Probability of receiving an incorrect cost
        self.prob_incorrect_reaction   = (1 - self.prob_selecting_correct_target_reaction)*self.prob_making_given_reaction*self.prob_selecting_reaction
        self.prob_incorrect_gamble     = (1 - self.prob_selecting_correct_target_gamble)*self.prob_making_given_gamble*self.prob_selecting_gamble
        self.prob_incorrect            = self.prob_incorrect_reaction + self.prob_incorrect_gamble
        
        #* Probability of receiving an indecision cost (No chance of success for indecision, so we just multiply by the two marginal probs calculated in last section)
        self.prob_indecision_reaction  = 1 - self.prob_making_reaction
        self.prob_indecision_gamble    = 1 - self.prob_making_gamble
        self.prob_indecision           = 1 - self.prob_making
                
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
        For exp2 with changing reward, just need agent_means and agent_sds to be length of 4 of the same numbers
        '''
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        output_check = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        for i in range(self.num_blocks):
            output[i,:] = stats.norm.cdf(self.timesteps[i,:],self.agent_means[i],self.agent_sds[i])
            output_check[i,:] = norm.cdf(self.timesteps[i,:],self.agent_means[i],self.agent_sds[i])
        
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
        combined_uncertainty = np.sqrt(self.timing_uncertainty**2 + self.agent_sds**2) # Prob of SELECTING only includes timing uncertainty and agent uncertainty
        
        # I've determined that the decision time just needs to be after, doesn't necessarily need to be after some decision action delay
        for i in range(self.num_blocks):   
            diff = self.timesteps[i,:] - self.agent_means[i]
            output[i,:] = 1 - stats.norm.cdf(self.weird_reaction_gamble_cutoff,diff,combined_uncertainty[i]) 
                
        return output
    
    def prob_making_for_reaction(self):
        # Get first three central moments (EX2 is normalized for mean, EX3 is normalized for mean and sd) of the new distribution based on timing uncertainty
        inf_timesteps = np.arange(0,5000,1,dtype=np.float64)
        time_means = self.timesteps[0,:]
        
        # Get the First Three moments for the left and right distributions (if X<Y and if X>Y respectively)
        self.EX_R,self.EX2_R,self.EX3_R,self.EX_G,self.EX2_G,self.EX3_G = get_moments(inf_timesteps,self.agent_means,time_means,self.agent_sds,self.timing_uncertainty)
        
        # Calculate the mean, variance, and skew with method of moments
        self.cutoff_agent_reaction_mean,self.cutoff_var,self.cutoff_skew = self.EX_R,get_variance(self.EX_R,self.EX2_R),get_skew(self.EX_R,self.EX2_R,self.EX3_R)
        self.cutoff_agent_reaction_sd = np.sqrt(self.cutoff_var)
        
        # same thing for gamble (X>Y)
        self.cutoff_agent_gamble_mean,self.cutoff_var,self.cutoff_skew = self.EX_G,get_variance(self.EX_G,self.EX2_G),get_skew(self.EX_G,self.EX2_G,self.EX3_G)
        self.cutoff_agent_gamble_sd = np.sqrt(self.cutoff_var)
        
        # Calculate the prob of making it on a reaction 
        #! Cutoff agent distribution isn't normal, so might not be able to simply add these, problem for later
        prob_make_reaction = stats.norm.cdf(1500,self.cutoff_agent_reaction_mean + self.reaction_plus_movement_time,np.sqrt(self.cutoff_agent_reaction_sd**2 + self.reaction_plus_movement_uncertainty**2))
        
        return prob_make_reaction
    
    def prob_making_for_gamble(self):
        self.gamble_reach_time_mean   = self.timesteps + self.movement_time + self.expected_gamble_delay
        self.gamble_reach_time_sd     = np.sqrt(self.expected_gamble_uncertainty**2 + self.movement_uncertainty**2)        
        output1 = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        
        for i in range(self.num_blocks): 
            output1[i,:] = stats.norm.cdf(1500,self.gamble_reach_time_mean[i,:],self.gamble_reach_time_sd[i])
        return output1
    
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
        self.optimal_expected_reaction_leave_target_time_sd   = np.sqrt(self.optimal_expected_cutoff_agent_reaction_sd**2 + self.reaction_uncertainty**2)
        # Find optimal gamble leave target time and sd
        self.optimal_expected_gamble_leave_target_time_mean = self.optimal_decision_time + self.expected_gamble_delay
        self.optimal_expected_gamble_leave_target_time_sd = np.sqrt(self.expected_gamble_uncertainty**2)
        
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
        
        ############################ TRUE ##########################################
        self.optimal_true_prob_selecting_reaction             = self.optimal_expected_prob_selecting_reaction
        self.optimal_true_prob_selecting_gamble             = self.optimal_expected_prob_selecting_gamble
        self.optimal_true_reaction_leave_target_time_mean     = self.optimal_expected_reaction_leave_target_time_mean
        self.optimal_true_reaction_leave_target_time_sd       = self.optimal_expected_reaction_leave_target_time_sd
        self.optimal_true_gamble_leave_target_time_mean       = self.optimal_decision_time + self.true_gamble_delay
        self.optimal_true_gamble_leave_target_time_sd         = np.sqrt(self.true_gamble_uncertainty**2)
        
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
                 
        return
    
    # * Calculate experiment metrics possibly with unknown gamble stuff
    def get_true_experiment_metrics(self):        
        if True:
            # ! Here we calculate the probability of making a reaction GIVEN we know that you selected reaction (conditional)
            # Prob of making it to the target
            self.optimal_true_prob_making_given_reaction,self.optimal_true_prob_making_given_gamble = self.optimal_true_prob_making_for_reaction_and_gamble()
            
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
            self.optimal_true_phat_correct = self.optimal_true_prob_selecting_reaction*1.0 + self.optimal_true_prob_selecting_gamble*0.5

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
            self.optimal_true_prob_indecision_reaction  = 1 - self.optimal_true_prob_making_reaction
            self.optimal_true_prob_indecision_gamble    = 1 - self.optimal_true_prob_making_gamble
            self.optimal_true_prob_indecision           = 1 - self.optimal_true_prob_making    
            
    def optimal_true_prob_making_for_reaction_and_gamble(self):
        '''
        Takes into account (un)known gamble delay
        '''
        # Reaction calc
        self.optimal_true_reaction_reach_time = self.optimal_expected_cutoff_agent_reaction_mean + self.reaction_plus_movement_time
        self.reaction_reach_time_uncertainty = np.sqrt(self.optimal_expected_cutoff_agent_reaction_sd**2 + self.reaction_plus_movement_uncertainty**2)
        output1 = stats.norm.cdf(1500,self.optimal_true_reaction_reach_time,self.reaction_reach_time_uncertainty)
        
        # Gamble Calc
        self.optimal_true_gamble_reach_time_mean = self.optimal_decision_time + self.true_gamble_delay + self.movement_time
        # ? Gamble uncertainty without the (un)known uncertainty includes agent as well
        # But also, the (un)known uncertainty from the data INCLUDES the agent bc people are experiencing the agent
        # How do we parse between people's switch uncertainty and the uncertainty in gamble times due to the agent?
        self.optimal_true_gamble_reach_time_uncertainty = np.sqrt(self.true_gamble_uncertainty**2 + self.movement_uncertainty**2) 
        output2 = stats.norm.cdf(1500,self.optimal_true_gamble_reach_time_mean,self.optimal_true_gamble_reach_time_uncertainty)
        return output1,output2         

    def calculate_gamble_reaction_metrics(self):
        self.temp_prob_win        = self.replace_zero_with_nan(self.optimal_true_prob_win)
        self.temp_prob_indecision = self.replace_zero_with_nan(self.optimal_true_prob_indecision)
        self.temp_prob_incorrect  = self.replace_zero_with_nan(self.optimal_true_prob_incorrect)
        if True:
            # Percent of metric that were reaction and gamble
            self.optimal_true_prob_wins_that_were_gamble          = (self.optimal_true_prob_win_gamble)/self.temp_prob_win
            self.optimal_true_prob_indecisions_that_were_gamble   = (self.optimal_true_prob_indecision_gamble)/self.temp_prob_indecision
            self.optimal_true_prob_incorrects_that_were_gamble    = (self.optimal_true_prob_incorrect_gamble)/self.temp_prob_incorrect
            
            self.optimal_true_prob_wins_that_were_reaction        = (self.optimal_true_prob_win_reaction)/self.temp_prob_win
            self.optimal_true_prob_indecisions_that_were_reaction = (self.optimal_true_prob_indecision_reaction)/self.temp_prob_indecision
            self.optimal_true_prob_incorrects_that_were_reaction  = (self.optimal_true_prob_incorrect_reaction)/self.temp_prob_incorrect
            
            # Probent of reaction or gamble that were wins/incorrects/indecisions
            self.optimal_true_prob_gambles_that_were_wins          = (self.optimal_true_prob_win_gamble)/self.optimal_expected_prob_selecting_gamble
            self.optimal_true_prob_gambles_that_were_incorrects    = (self.optimal_true_prob_incorrect_gamble)/self.optimal_expected_prob_selecting_gamble
            self.optimal_true_prob_gambles_that_were_indecisions   = (self.optimal_true_prob_indecision_gamble)/self.optimal_expected_prob_selecting_gamble
            
            self.optimal_true_prob_reactions_that_were_wins        = (self.optimal_true_prob_win_reaction)/self.optimal_expected_prob_selecting_reaction
            self.optimal_true_prob_reactions_that_were_incorrects  = (self.optimal_true_prob_incorrect_reaction)/self.optimal_expected_prob_selecting_reaction
            self.optimal_true_prob_reactions_that_were_indecisions = (self.optimal_true_prob_indecision_reaction)/self.optimal_expected_prob_selecting_reaction
    
    
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
        
    def mseloss(self,decision_time):
        # Go through the model with these specific DECISION times
        self.calculate_metrics_with_certain_decision_time(decision_time)
        
        # Get wins,indecisions,incorrects,and leave target times and compare to data
        win_diff                 = abs(self.optimal_true_prob_win*100 - self.tune_data[0])
        indecision_diff          = abs(self.optimal_true_prob_indecision*100 - self.tune_data[1])
        incorrect_diff           = abs(self.optimal_true_prob_incorrect*100 - self.tune_data[2])
        leave_target_time_diff   = abs(self.wtd_optimal_true_leave_target_time - self.tune_data[3])
        perc_reactions_diff      = abs(self.optimal_true_prob_selecting_reaction*100 - self.tune_data[4])
        perc_gambles_diff        = abs(self.optimal_true_prob_selecting_gamble*100 - self.tune_data[5])
        reaction_leave_time_diff = abs(self.optimal_true_reaction_leave_target_time_mean - self.tune_data[6])
        gamble_leave_time_diff   = abs(self.optimal_true_gamble_leave_target_time_mean - self.tune_data[7])
        
        metric_loss = np.array([win_diff,indecision_diff,incorrect_diff,leave_target_time_diff,
                                perc_reactions_diff,perc_gambles_diff,reaction_leave_time_diff,gamble_leave_time_diff])
        return metric_loss 
    
    def fit_model_to_data(self,data):
        '''
        data = [wins,indecisions,incorrects,decision_times,perc_reaction_decisions,perc_gamble_decisions]
        '''
        self.tune_data      = data
        self.tune_timesteps = np.arange(900,1800,1)
        decision_times      = np.array([self.tune_timesteps[0]]*self.num_blocks) # Start off with 600 for each parameter
        num_metrics         = len(self.tune_data)
        loss_store          = np.zeros((num_metrics,self.num_blocks,len(self.tune_timesteps))) # Each metric,each block, each timestep
        
        for i in range(self.num_blocks):
            for j,t in enumerate(self.tune_timesteps):
                decision_times[i] = t
                metric_loss = self.mseloss(decision_times)
                loss_store[:,i,j] = metric_loss[:,i]
                
        self.fit_decision_times = np.zeros((num_metrics,self.num_blocks))
        for i in range(num_metrics):
            for j in range(self.num_blocks):
                self.fit_decision_times[i,j] = np.argmin(loss_store[i,j,:]) + np.min(self.tune_timesteps)
        self.fit_decision_times_dict = {'Wins':self.fit_decision_times[0,:],'Indecisions':self.fit_decision_times[1,:],
                                        'Incorrects':self.fit_decision_times[2,:],'Leave Target Time':self.fit_decision_times[3,:],
                                        'Perc Reaction Decisions':self.fit_decision_times[4,:],'Perc Gamble Decisions':self.fit_decision_times[5,:],
                                        'Reaction Leave Time': self.fit_decision_times[6,:],'Gamble Leave Time': self.fit_decision_times[7,:]}    
        
@njit(parallel=True)
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