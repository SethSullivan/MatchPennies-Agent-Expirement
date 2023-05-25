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
    ans = (EX3 - 3*EX*(EX2 - EX**2) - EX**3)/((EX2 - EX**2)**(3/2))
    return ans
def get_variance(EX,EX2):
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
        # * MODEL VARIATION PARAMETERS ON/OFF
        self.unknown_gamble_uncertainty_on = kwargs.get('unknown_gamble_uncertainty_on',False)
        self.unknown_gamble_delay_on       = kwargs.get('unknown_gamble_delay_on',False)
        self.known_gamble_uncertainty_on   = kwargs.get('known_gamble_uncertainty_on',False)
        self.known_gamble_delay_on         = kwargs.get('known_gamble_delay_on',False)
        # * Model Variation Parameters:
        if True:
            self.unknown_gamble_uncertainty   = kwargs.get('unknown_gamble_uncertainty')
            self.unknown_gamble_delay         = kwargs.get('unknown_gamble_delay')
            self.known_gamble_uncertainty     = kwargs.get('known_gamble_uncertainty')
            self.known_gamble_delay           = kwargs.get('known_gamble_delay')
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
            self.decision_action_delay_uncertainty  = kwargs.get('decision_action_delay_uncertainty')
            self.reaction_plus_movement_uncertainty = np.sqrt(self.reaction_uncertainty**2 + self.movement_uncertainty**2)
            self.total_uncertainty                  = np.sqrt(self.reaction_plus_movement_uncertainty**2 + self.timing_uncertainty**2)
            self.total_uncertainty_reaction         = self.reaction_plus_movement_uncertainty
            self.total_uncertainty_gamble           = self.movement_uncertainty 
            self.agent_plus_human_uncertainty       = np.sqrt(self.total_uncertainty**2 + self.agent_sds**2)
            # Ability
            self.reaction_time               = kwargs.get('reaction_time')
            self.movement_time               = kwargs.get('movement_time')
            self.reaction_plus_movement_time = self.reaction_time + self.movement_time
            self.decision_action_delay_mean  = kwargs.get('decision_action_delay_mean')
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

    def mseloss(self,decision_time):
        # Go through the model with these specific DECISION times
        self.calculate_metrics_with_certain_decision_time(decision_time)
        
        # Get wins,indecisions,incorrects,and leave target times and compare to data
        win_diff                 = abs(self.perc_win_optimal_calc - self.tune_data[0])
        indecision_diff          = abs(self.perc_indecision_optimal_calc - self.tune_data[1])
        incorrect_diff           = abs(self.perc_incorrect_optimal_calc - self.tune_data[2])
        leave_target_time_diff   = abs(self.wtd_optimal_leave_target_time - self.tune_data[3])
        perc_reactions_diff      = abs(self.prob_selecting_reaction_optimal_calc*100 - self.tune_data[4])
        perc_gambles_diff        = abs(self.prob_selecting_gamble_optimal_calc*100 - self.tune_data[5])
        reaction_leave_time_diff = abs(self.optimal_reaction_leave_target_time_mean_calc - self.tune_data[6])
        gamble_leave_time_diff   = abs(self.optimal_gamble_leave_target_time_mean_calc - self.tune_data[7])
        
        metric_loss = np.array([win_diff,indecision_diff,incorrect_diff,leave_target_time_diff,
                                perc_reactions_diff,perc_gambles_diff,reaction_leave_time_diff,gamble_leave_time_diff])
        return metric_loss 
       
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
            # ax.set_ylim(np.min(self.incorrect_cost,self.indecision_cost)-0.03,np.max(self.win_reward)+0.03)
            ax.set_xlim(0,1500)
            ax.set_xticks(np.arange(0,2000,300))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Expected Reward')
            ax.legend(fontsize = 8,loc = (0.01,0.1))
            ax.set_title(f'Gain Function for Decision Time\nAgent Mean,SD = {self.agent_means[i]},{self.agent_sds[i]}')#\n B = {B}')
            plt.show()  
    
    
    #################################################################### 
    ################### ------ Run Model -----#####################
    #################################################################### 
    def run_model(self):
        ###################### ----- Find Expected Reward Every Timestep -----  ######################
        # * Probabilites ( Run through Functions)
        if True:
            #* Agent probabilities (not used, agent behavior is used in prob_of_selecting_reaction)
            self.prob_agent_has_gone     = self.prob_agent_go()
            self.prob_agent_has_not_gone = 1 - self.prob_agent_has_gone
            
            #* Prob of selecting reacting or gambling decision
            self.prob_selecting_reaction = self.prob_of_selecting_reaction() # Probability of SELECTING a Decision only depends on timing uncertainty, not total uncertainty
            self.prob_selecting_gamble   = 1 - self.prob_selecting_reaction
            
            # * Here we calculate the probability of making a reaction GIVEN we know that you selected reaction (conditional)
            # * THis uses the truncated agent distribution to determine if you'll make it or not
            # Prob of making it to the target
            self.prob_making_given_reaction = self.prob_making_for_reaction()
            self.prob_making_given_gamble   = self.prob_making_for_gamble()
            
            # Prob of win
            self.prob_win_given_reaction = self.prob_selecting_correct_target_reaction*self.prob_making_given_reaction
            self.prob_win_given_gamble = self.prob_selecting_correct_target_gamble*self.prob_making_given_gamble
            
            # Prob of indecision
            self.prob_indecision_given_reaction = 1 - self.prob_making_given_reaction
            self.prob_indecision_given_gamble = 1 - self.prob_making_given_gamble
            
            # Prob of incorrect
            self.prob_incorrect_given_reaction = (1 - self.prob_selecting_correct_target_reaction)*self.prob_making_given_reaction
            self.prob_incorrect_given_gamble = (1 - self.prob_selecting_correct_target_gamble)*self.prob_making_given_gamble
            
            
        # * Prob of win, incorrect, indecisions (All equations, no functions)
        if True: 
            # * Prob making on reaction and gamble depends on the prob of selecting reaction and gamble too
            self.prob_making_reaction  = self.prob_making_given_reaction*self.prob_selecting_reaction
            self.prob_making_gamble    = self.prob_making_given_gamble*self.prob_selecting_gamble
            self.prob_making           = self.prob_making_gamble + self.prob_making_reaction
            
            #* Multiply the actual probability of making it times the prob of getting it right for reaction and gamble
            self.prob_win_reaction = self.prob_selecting_correct_target_reaction*self.prob_making_reaction
            self.prob_win_gamble   = self.prob_selecting_correct_target_gamble*self.prob_making_gamble
            self.prob_win          = self.prob_win_reaction + self.prob_win_gamble
            
            #* Probability of receiving an incorrect cost
            self.prob_incorrect_reaction = self.prob_incorrect_given_reaction*self.prob_making_reaction
            self.prob_incorrect_gamble   = self.prob_incorrect_given_gamble*self.prob_making_gamble
            self.prob_incorrect          = self.prob_incorrect_reaction + self.prob_incorrect_gamble
            
            #* Probability of receiving an indecision cost (No chance of success for indecision, so we just multiply by the two marginal probs calculated in last section)
            self.prob_indecision_reaction = 1 - self.prob_making_reaction
            self.prob_indecision_gamble   = 1 - self.prob_making_gamble
            self.prob_indecision          = 1 - self.prob_making
        # * Expected reward calculation 
        if True:
            self.exp_reward_reaction = (self.prob_win_reaction*self.win_reward + self.prob_incorrect_reaction*self.incorrect_cost + self.prob_indecision_reaction*self.indecision_cost)
            self.exp_reward_gamble   = (self.prob_win_gamble*self.win_reward + self.prob_incorrect_gamble*self.incorrect_cost + self.prob_indecision_gamble*self.indecision_cost )
            self.exp_reward1         = self.exp_reward_reaction + self.exp_reward_gamble
            self.exp_reward          = self.prob_win*self.win_reward + self.prob_incorrect*self.incorrect_cost + self.prob_indecision*self.indecision_cost 
        
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

        ###################### ----- Get Experiment Metrics ----- #####################
                
        # * Calculate experiment metrics from expected reward 
        # (uses optimal index on all the calculations we already made to find the optimal)
        self.get_experiment_metrics_from_expected_reward()
        
        # * Find the experiment metrics probabilistically 
        # (this is useful when calculating the wins people ACTUALLY get when there is something unknown
        # in the calculation of the gain function)
        self.calculate_experiment_metrics()
        
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
        on this, so the timing uncertainty is included
        
        The uncertainty here depends on if people have knowledge of their gamble uncertainty or not
            - Gamble uncertainty is larger than peoples timing uncertainty
            - I'm not sure if gamble uncertainty should decide your chances of selecting reaction or not (as of 2/21/23 I'm saying yes it makes a difference)
        '''
        # ! AS OF 4/04/23 I think gamble uncertainty should not affect the prob of selecting reaction or gamble
        # 4/06/23 - GAMBLE UNCERTAINTY from the data is a product of both my own timing uncertainty and the agents uncertainty
        # ! Should compare the combination of these two things to see if they match the gamble uncertainty in the data
        # I think the gamble uncertainty is the uncertainty in the time it takes you to go from decision to movement and when you switch it's really uncertain 

        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        # if self.known_gamble_uncertainty_on:
        #     # IF it's an array, i'm using data and therefore agent SD is already included
        #     if isinstance(self.known_gamble_uncertainty,np.ndarray):
        #         combined_uncertainty = np.sqrt(self.known_gamble_uncertainty**2)# + self.decision_action_delay_uncertainty**2) # Prob of SELECTING only includes timing uncertainty and agent uncertainty
        #         combined_uncertainty = np.tile(combined_uncertainty,(2000,1)).T
        #     # If I'm using one number 
        #     else:
        #         combined_uncertainty = np.sqrt(self.known_gamble_uncertainty**2 + self.agent_sds**2)# + self.decision_action_delay_uncertainty**2)
        # else:
        combined_uncertainty = np.sqrt(self.timing_uncertainty**2 + self.agent_sds**2)# + self.decision_action_delay_uncertainty**2) # Prob of SELECTING only includes timing uncertainty and agent uncertainty
        
        # I've determined that the decision time just needs to be after, doesn't necessarily need to be after some decision action delay
        for i in range(self.num_blocks):   
            diff = self.timesteps[i,:] - self.agent_means[i]
            output[i,:] = 1 - stats.norm.cdf(self.weird_reaction_gamble_cutoff,diff,combined_uncertainty[i]) # I've determined that the decision time just needs to be after, doesn't necessarily need to be after some decision action delay
                
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
        prob_make_reaction = stats.norm.cdf(1500,self.cutoff_agent_reaction_mean + self.reaction_plus_movement_time,np.sqrt(self.cutoff_agent_reaction_sd**2 + self.reaction_plus_movement_uncertainty**2))
        
        return prob_make_reaction
    
    def prob_making_for_gamble(self):
        if self.known_gamble_delay_on:
            self.gamble_reach_time_mean = self.timesteps + self.movement_time + self.known_gamble_delay # Gamble delay includes decision action delay, but there's another cognitive delay involved
        else:
            self.gamble_reach_time_mean = self.timesteps + self.movement_time + self.decision_action_delay_mean
        
        if self.known_gamble_uncertainty_on:
            self.gamble_reach_time_uncertainty = np.sqrt(self.known_gamble_uncertainty**2 + self.movement_uncertainty**2)
        else:
            self.gamble_reach_time_uncertainty = np.sqrt(self.timing_uncertainty**2 + self.movement_uncertainty**2) # The timing uncertainy measure in coincidence task includes 
        
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        for i in range(self.num_blocks): 
            output[i,:] = stats.norm.cdf(1500,self.gamble_reach_time_mean[i,:],self.gamble_reach_time_uncertainty[i])
        
        return output
    
    ########################################################################
    ##########---- Functions to Calculate Optimal Metrics ------############
    ########################################################################
    
    # * Just get the probabilities from the ER calculations, 
    # these are the same as calculating the experiment metrics post-hoc, but those ones
    # can include the unknown shit
    def get_experiment_metrics_from_expected_reward(self):
        '''
        Get the index of the calculated probabilities
        
        This won't be correct if there are unknown uncertainties
        '''
        if self.BETA_ON:
            self.ER_index = self.beta_optimal_index
        else:
            self.ER_index = self.optimal_index
        self.prob_selecting_reaction_optimal_ER        = np.zeros(self.num_blocks)
        self.prob_selecting_gamble_optimal_ER          = np.zeros(self.num_blocks)
        self.prob_making_reaction_optimal_ER           = np.zeros(self.num_blocks)
        self.prob_making_gamble_optimal_ER             = np.zeros(self.num_blocks)
        self.prob_making_given_reaction_optimal_ER     = np.zeros(self.num_blocks)
        self.prob_making_given_gamble_optimal_ER       = np.zeros(self.num_blocks)
        self.prob_indecision_gamble_optimal_ER         = np.zeros(self.num_blocks)
        self.prob_indecision_reaction_optimal_ER       = np.zeros(self.num_blocks)
        self.prob_indecision_given_gamble_optimal_ER   = np.zeros(self.num_blocks)
        self.prob_indecision_given_reaction_optimal_ER = np.zeros(self.num_blocks)
        self.prob_incorrect_gamble_optimal_ER          = np.zeros(self.num_blocks)
        self.prob_incorrect_reaction_optimal_ER        = np.zeros(self.num_blocks)
        self.prob_incorrect_given_gamble_optimal_ER    = np.zeros(self.num_blocks)
        self.prob_incorrect_given_reaction_optimal_ER  = np.zeros(self.num_blocks)
        self.prob_win_gamble_optimal_ER                = np.zeros(self.num_blocks)
        self.prob_win_reaction_optimal_ER              = np.zeros(self.num_blocks)
        self.prob_win_given_gamble_optimal_ER          = np.zeros(self.num_blocks)
        self.prob_win_given_reaction_optimal_ER        = np.zeros(self.num_blocks)
        self.prob_win_optimal_ER                       = np.zeros(self.num_blocks)
        self.prob_indecision_optimal_ER                = np.zeros(self.num_blocks)
        self.prob_incorrect_optimal_ER                 = np.zeros(self.num_blocks)
        for i in range(self.num_blocks):
            self.prob_making_given_reaction_optimal_ER[i]     = self.prob_making_given_reaction[i,self.ER_index[i]]
            self.prob_making_given_gamble_optimal_ER[i]       = self.prob_making_given_gamble[i,self.ER_index[i]]
            self.prob_making_reaction_optimal_ER[i]           = self.prob_making_reaction[i,self.ER_index[i]]
            self.prob_making_gamble_optimal_ER[i]             = self.prob_making_gamble[i,self.ER_index[i]]
            
            self.prob_selecting_reaction_optimal_ER[i]        = self.prob_selecting_reaction[i,self.ER_index[i]]
            self.prob_selecting_gamble_optimal_ER[i]          = self.prob_selecting_gamble[i,self.ER_index[i]]
            
            self.prob_indecision_given_gamble_optimal_ER[i]   = self.prob_indecision_given_gamble[i,self.ER_index[i]]
            self.prob_indecision_given_reaction_optimal_ER[i] = self.prob_indecision_given_reaction[i,self.ER_index[i]]
            self.prob_indecision_gamble_optimal_ER[i]         = self.prob_indecision_given_gamble[i,self.ER_index[i]]
            self.prob_indecision_reaction_optimal_ER[i]       = self.prob_indecision_given_reaction[i,self.ER_index[i]]
            
            self.prob_incorrect_given_gamble_optimal_ER[i]    = self.prob_incorrect_given_gamble[i,self.ER_index[i]]
            self.prob_incorrect_given_reaction_optimal_ER[i]  = self.prob_incorrect_given_reaction[i,self.ER_index[i]]
            self.prob_incorrect_gamble_optimal_ER[i]          = self.prob_incorrect_gamble[i,self.ER_index[i]]
            self.prob_incorrect_reaction_optimal_ER[i]        = self.prob_incorrect_reaction[i,self.ER_index[i]]
            
            self.prob_win_given_gamble_optimal_ER[i]          = self.prob_win_given_gamble[i,self.ER_index[i]]
            self.prob_win_given_reaction_optimal_ER[i]        = self.prob_win_given_reaction[i,self.ER_index[i]]
            self.prob_win_gamble_optimal_ER[i]                = self.prob_win_gamble[i,self.ER_index[i]]
            self.prob_win_reaction_optimal_ER[i]              = self.prob_win_reaction[i,self.ER_index[i]]
            
            self.prob_win_optimal_ER[i]                       = self.prob_win[i,self.ER_index[i]]
            self.prob_incorrect_optimal_ER[i]                 = self.prob_incorrect[i,self.ER_index[i]]
            self.prob_indecision_optimal_ER[i]                = self.prob_indecision[i,self.ER_index[i]]
    
    
    # * Calculate experiment metrics possibly with unknown gamble stuff
    def calculate_experiment_metrics(self):
        # * Set gamble delay
        self.set_gamble_delay_and_uncertainty()
        
         # * Find probability of selecting gamble/reactions
        self.calculate_prob_of_selecting_reaction_or_gamble()
        
        # * Calculate the mean leave target time based on the decision time
        # Calculates reaction and gamble leave target times too
        self.wtd_optimal_leave_target_time,self.wtd_optimal_reach_target_time,self.predicted_decision_time = self.calculate_mean_leave_target_time()
        
        self.player_minus_agent_leave_time          = self.wtd_optimal_leave_target_time - self.agent_means
        self.player_minus_agent_reaction_leave_time = self.optimal_reaction_leave_target_time_mean_calc - self.cutoff_agent_reaction_mean_optimal_ER
        self.player_minus_agent_gamble_leave_time   = self.optimal_gamble_leave_target_time_mean_calc - self.cutoff_agent_gamble_mean_optimal_ER
        
        if True:
            # ! Here we calculate the probability of making a reaction GIVEN we know that you selected reaction (conditional)

            # Prob of making it to the target
            self.prob_making_given_reaction_optimal_calc,self.prob_making_given_gamble_optimal_calc = self.prob_making_for_reaction_and_gamble_optimal_calc()
            
            # Prob of win
            self.prob_win_given_reaction_optimal_calc = self.prob_selecting_correct_target_reaction*self.prob_making_given_reaction_optimal_calc
            self.prob_win_given_gamble_optimal_calc = self.prob_selecting_correct_target_gamble*self.prob_making_given_gamble_optimal_calc
            
            # Prob of indecision
            self.prob_indecision_given_reaction_optimal_calc = 1 - self.prob_making_given_reaction_optimal_calc
            self.prob_indecision_given_gamble_optimal_calc = 1 - self.prob_making_given_gamble_optimal_calc
            
            # Prob of incorrect
            self.prob_incorrect_given_reaction_optimal_calc = (1 - self.prob_selecting_correct_target_reaction)*self.prob_making_given_reaction_optimal_calc
            self.prob_incorrect_given_gamble_optimal_calc = (1 - self.prob_selecting_correct_target_gamble)*self.prob_making_given_gamble_optimal_calc
            
            # Doesn't matter if you make it, what's the probability you intend to go in the correct direction
            self.phat_correct_optimal_calc = self.prob_selecting_reaction_optimal_calc*1.0 + self.prob_selecting_gamble_optimal_calc*0.5

        # * Prob of win, incorrect, indecisions (All equations, no functions)
        if True:

            self.prob_making_reaction_optimal_calc       = self.prob_making_given_reaction_optimal_calc*self.prob_selecting_reaction_optimal_calc #! THIS INCLUDES PROB SELECTING... DON'T MULTIPLY AGAIN
            self.prob_making_gamble_optimal_calc         = self.prob_making_given_gamble_optimal_calc*self.prob_selecting_gamble_optimal_calc
            self.prob_making_optimal_calc                = self.prob_making_reaction_optimal_calc + self.prob_making_gamble_optimal_calc
            
            self.prob_not_making_reaction_optimal_calc       = (1-self.prob_making_given_reaction_optimal_calc)*self.prob_selecting_reaction_optimal_calc #! THIS INCLUDES PROB SELECTING... DON'T MULTIPLY AGAIN
            self.prob_not_making_gamble_optimal_calc         = (1-self.prob_making_given_gamble_optimal_calc)*self.prob_selecting_gamble_optimal_calc
            self.prob_not_making_optimal_calc                = self.prob_not_making_reaction_optimal_calc + self.prob_not_making_gamble_optimal_calc
            
            # Multiply the actual probability of making it times the prob of getting it right for reaction and gamble
            self.prob_win_reaction_optimal_calc = self.prob_selecting_correct_target_reaction*self.prob_making_reaction_optimal_calc
            self.prob_win_gamble_optimal_calc   = self.prob_selecting_correct_target_gamble*self.prob_making_gamble_optimal_calc
            self.prob_win_optimal_calc          = self.prob_win_reaction_optimal_calc + self.prob_win_gamble_optimal_calc
            self.perc_win_optimal_calc          = self.prob_win_optimal_calc*100

            # Probability of receiving an incorrect cost
            self.prob_incorrect_reaction_optimal_calc = self.prob_incorrect_given_reaction_optimal_calc*self.prob_making_reaction_optimal_calc
            self.prob_incorrect_gamble_optimal_calc   = self.prob_incorrect_given_gamble_optimal_calc*self.prob_making_gamble_optimal_calc
            self.prob_incorrect_optimal_calc          = self.prob_incorrect_reaction_optimal_calc + self.prob_incorrect_gamble_optimal_calc
            self.perc_incorrect_optimal_calc          = self.prob_incorrect_optimal_calc*100
            # Probability of receiving an indecision cost (No chance of success for indecision, so we just multiply by the two marginal probs calculated in last section)
            self.prob_indecision_reaction_optimal_calc  = self.prob_indecision_given_reaction_optimal_calc*self.prob_selecting_reaction_optimal_calc
            self.prob_indecision_gamble_optimal_calc    = self.prob_indecision_given_gamble_optimal_calc*self.prob_selecting_gamble_optimal_calc
            self.prob_indecision_optimal_calc           = self.prob_indecision_reaction_optimal_calc + self.prob_indecision_gamble_optimal_calc
            self.perc_indecision_optimal_calc           = self.prob_indecision_optimal_calc*100

            self.calculate_gamble_reaction_metrics()     
        
    def set_gamble_delay_and_uncertainty(self):
        if self.unknown_gamble_delay_on:
            self.gamble_delay = self.unknown_gamble_delay
        elif self.known_gamble_delay_on:
            self.gamble_delay = self.known_gamble_delay
        else:
            self.gamble_delay = self.decision_action_delay_mean
            
        if self.unknown_gamble_uncertainty_on:
            self.gamble_uncertainty   = self.unknown_gamble_uncertainty
        elif self.known_gamble_uncertainty_on:
            self.gamble_uncertainty   = self.known_gamble_uncertainty
        else:
            # Using total agent uncertainty, as opposed to the cutoff uncertainty, not sure which makes more sense
            self.gamble_uncertainty   = np.sqrt(self.timing_uncertainty**2) # 
 
    def calculate_prob_of_selecting_reaction_or_gamble(self):
        '''

        '''
        self.prob_selecting_reaction_optimal_calc = np.zeros(self.num_blocks)
        self.prob_selecting_gamble_optimal_calc = np.zeros(self.num_blocks)
        # Only reason to recalculate this is if the gamble uncertainty is unknown... if it's known or None, then we go to the else, which can just use the calc during expected reward
        
        # for i in range(6):
        #     combined_uncertainty = np.sqrt(self.timing_uncertainty[i]**2 + self.agent_sds[i]**2)
        #     diff = self.optimal_index[i] - self.agent_means[i]
        #     self.prob_selecting_reaction_optimal_calc[i] = 1 - stats.norm.cdf(self.weird_reaction_gamble_cutoff,diff,combined_uncertainty) # I've determined that the decision time just needs to be after, doesn't necessarily need to be after some decision action delay
        combined_uncertainty = np.sqrt(self.timing_uncertainty**2 + self.agent_sds**2)
        diff = self.optimal_index - self.agent_means
        self.prob_selecting_reaction_optimal_calc = 1 - stats.norm.cdf(self.weird_reaction_gamble_cutoff,diff,combined_uncertainty) #
        self.prob_selecting_gamble_optimal_calc = 1 - self.prob_selecting_reaction_optimal_calc
            
        #Get into percentage 
        self.percent_reactions_optimal_calc = self.prob_selecting_reaction_optimal_calc*100
        self.percent_gambles_optimal_calc = self.prob_selecting_gamble_optimal_calc*100
        # self.percent_making_reaction_optimal = self.prob_making_reaction_optimal*100
        # self.percent_making_gamble_optimal = self.prob_making_given_gamble_optimal*100
            
    def calculate_mean_leave_target_time(self):
        # ! The cutoff agent reaction/gamble mean is unaffected by any of the unknown parameters
        # Unknown gamble delay and gamble uncertainty shouldn't theoretically affect the cutoff ability of the players
        # Get the agent's mean and sd for reaction and gambles at the optimal cutoff time
        self.cutoff_agent_reaction_mean_optimal_ER = np.zeros(self.num_blocks)
        self.cutoff_agent_reaction_sd_optimal_ER = np.zeros(self.num_blocks)
        self.cutoff_agent_gamble_mean_optimal_ER = np.zeros(self.num_blocks)
        self.cutoff_agent_gamble_sd_optimal_ER = np.zeros(self.num_blocks)
        for i in range(self.num_blocks):
            self.cutoff_agent_reaction_mean_optimal_ER[i] = self.cutoff_agent_reaction_mean[i,self.optimal_index[i]]
            self.cutoff_agent_reaction_sd_optimal_ER[i] = self.cutoff_agent_reaction_sd[i,self.optimal_index[i]]
            self.cutoff_agent_gamble_mean_optimal_ER[i] = self.cutoff_agent_gamble_mean[i,self.optimal_index[i]]
            self.cutoff_agent_gamble_sd_optimal_ER[i] = self.cutoff_agent_gamble_sd[i,self.optimal_index[i]]
        
        # Optimal reaction leave target time is agent mean + reaction time
        self.optimal_reaction_leave_target_time_mean_calc = self.cutoff_agent_reaction_mean_optimal_ER + self.reaction_time
        self.optimal_reaction_leave_target_time_sd_calc   = np.sqrt(self.cutoff_agent_reaction_sd_optimal_ER**2 + self.reaction_uncertainty**2)
        
        # Find optimal gamble leave target time and sd
        self.optimal_gamble_leave_target_time_mean_calc    = self.optimal_decision_time + self.gamble_delay
        self.optimal_gamble_leave_target_time_sd_calc      = np.sqrt(self.gamble_uncertainty**2 + self.cutoff_agent_gamble_sd_optimal_ER**2)
        
        # Get the leave target time by weighing by how often they react and gamble
        wtd_optimal_leave_target_time = (self.prob_selecting_reaction_optimal_calc*self.optimal_reaction_leave_target_time_mean_calc + \
                                            self.prob_selecting_gamble_optimal_calc*self.optimal_gamble_leave_target_time_mean_calc)/(self.prob_selecting_gamble_optimal_calc+self.prob_selecting_reaction_optimal_calc) 
        self.wtd_optimal_sd_leave_target_time = np.sqrt(self.optimal_reaction_leave_target_time_sd_calc**2+self.optimal_gamble_leave_target_time_sd_calc**2)
        
        # Just add movement time for reach target time
        wtd_optimal_reach_target_time = wtd_optimal_leave_target_time + self.movement_time
        
        # Predict their decision time
        predicted_decision_time = (self.prob_selecting_reaction_optimal_calc*self.cutoff_agent_reaction_mean_optimal_ER) + \
                                       (self.prob_selecting_gamble_optimal_calc*(self.optimal_gamble_leave_target_time_mean_calc - self.gamble_delay))
            
        return wtd_optimal_leave_target_time,wtd_optimal_reach_target_time,predicted_decision_time
    
    def prob_making_for_reaction_and_gamble_optimal_calc(self):
        '''
        Takes into account (un)known gamble delay
        '''
        # Reaction calc
        self.reaction_reach_time_optimal_calc = self.cutoff_agent_reaction_mean_optimal_ER + self.reaction_plus_movement_time
        self.reaction_reach_time_uncertainty = np.sqrt(self.cutoff_agent_reaction_sd_optimal_ER**2 + self.reaction_plus_movement_uncertainty**2)
        output1 = stats.norm.cdf(1500,self.reaction_reach_time_optimal_calc,self.reaction_reach_time_uncertainty)
        
        # Gamble Calc
        self.gamble_reach_time_mean = self.optimal_decision_time + self.gamble_delay + self.movement_time
        # ? Gamble uncertainty without the (un)known uncertainty includes agent as well
        # But also, the (un)known uncertainty from the data INCLUDES the agent bc people are experiencing the agent
        # How do we parse between people's switch uncertainty and the uncertainty in gamble times due to the agent?
        self.gamble_reach_time_uncertainty = np.sqrt(self.gamble_uncertainty**2 + self.movement_uncertainty**2) 
        output2 = stats.norm.cdf(1500,self.gamble_reach_time_mean,self.gamble_reach_time_uncertainty)
        
        return output1,output2         

    def calculate_gamble_reaction_metrics(self):
        self.temp_prob_win        = self.replace_zero_with_nan(self.prob_win_optimal_calc)
        self.temp_prob_indecision = self.replace_zero_with_nan(self.prob_indecision_optimal_calc)
        self.temp_prob_incorrect  = self.replace_zero_with_nan(self.prob_incorrect_optimal_calc)
        if True:
            # Percent of metric that were reaction and gamble
            self.perc_wins_that_were_gamble_optimal_calc          = ((self.prob_win_gamble_optimal_calc)/self.temp_prob_win)*100
            self.perc_indecisions_that_were_gamble_optimal_calc   = ((self.prob_indecision_gamble_optimal_calc)/self.temp_prob_indecision)*100
            self.perc_incorrects_that_were_gamble_optimal_calc    = ((self.prob_incorrect_gamble_optimal_calc)/self.temp_prob_incorrect)*100
            
            self.perc_wins_that_were_reaction_optimal_calc        = ((self.prob_win_reaction_optimal_calc)/self.temp_prob_win)*100
            self.perc_indecisions_that_were_reaction_optimal_calc = ((self.prob_indecision_reaction_optimal_calc)/self.temp_prob_indecision)*100
            self.perc_incorrects_that_were_reaction_optimal_calc  = ((self.prob_incorrect_reaction_optimal_calc)/self.temp_prob_incorrect)*100
            
            # Percent of reaction or gamble that were wins/incorrects/indecisions
            self.perc_gambles_that_were_wins_optimal_calc          = ((self.prob_win_gamble_optimal_calc)/self.prob_selecting_gamble_optimal_calc)*100
            self.perc_gambles_that_were_incorrects_optimal_calc    = ((self.prob_incorrect_gamble_optimal_calc)/self.prob_selecting_gamble_optimal_calc)*100
            self.perc_gambles_that_were_indecisions_optimal_calc   = ((self.prob_indecision_gamble_optimal_calc)/self.prob_selecting_gamble_optimal_calc)*100
            
            self.perc_reactions_that_were_wins_optimal_calc        = ((self.prob_win_reaction_optimal_calc)/self.prob_selecting_reaction_optimal_calc)*100
            self.perc_reactions_that_were_incorrects_optimal_calc  = ((self.prob_incorrect_reaction_optimal_calc)/self.prob_selecting_reaction_optimal_calc)*100
            self.perc_reactions_that_were_indecisions_optimal_calc = ((self.prob_indecision_reaction_optimal_calc)/self.prob_selecting_reaction_optimal_calc)*100
    
    
    ###########################################################
    #############----- Fit Model Functions ---- ###############
    ###########################################################
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

    def calculate_metrics_with_certain_decision_time(self,decision_times,final = False):
        self.optimal_decision_time = decision_times
        self.expected_reward_of_decision_time = np.zeros((self.num_blocks))
        if final:
            for i in range(self.num_blocks):
                self.expected_reward_of_decision_time[i] = self.exp_reward[i,int(self.optimal_decision_time[i])]
        self.optimal_index = (decision_times/self.nsteps).astype(int)

        self.get_experiment_metrics_from_expected_reward()
        self.calculate_experiment_metrics()
        
@njit(parallel=True)
def find_optimal_decision_time_for_certain_metric(ob,metric_name = 'RPMT'):
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