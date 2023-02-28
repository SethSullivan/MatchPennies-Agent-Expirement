import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import data_visualization as dv
wheel = dv.ColorWheel()
'''
This model finds the decision time that best fits the data for the group and the individual

This is a good check to see if the model actually captures everything that's going on 




'''
class Fit_Decision_Time_Model():
    def __init__(self, **kwargs):
        # Task Conditions
        self.num_blocks = kwargs.get('num_blocks',6)
        self.agent_means = kwargs.get('agent_means',np.array([1000,1000,1100,1100,1200,1200]))
        self.agent_stds = kwargs.get('agent_stds',np.array([50,150,50,150,50,150]))
        self.nsteps = 1
        self.timesteps = kwargs.get('timesteps',np.tile(np.arange(0,2000,self.nsteps),(self.num_blocks,1)))
        self.neg_inf_cut_off_value = -100000
        # MODEL VARIATION PARAMETERS ON/OFF
        self.unknown_gamble_uncertainty_on = kwargs.get('unknown_gamble_uncertainty_on',False)
        self.unknown_gamble_delay_on = kwargs.get('unknown_gamble_delay_on',False)
        self.known_gamble_uncertainty_on = kwargs.get('known_gamble_uncertainty_on',False)
        self.known_gamble_delay_on = kwargs.get('known_gamble_delay_on',False)
        # Model Variation Parameters:
        if True:
            self.unknown_gamble_uncertainty = kwargs.get('unknown_gamble_uncertainty')
            self.unknown_gamble_delay = kwargs.get('unknown_gamble_delay')
            self.known_gamble_uncertainty = kwargs.get('known_gamble_uncertainty')
            self.known_gamble_delay = kwargs.get('known_gamble_delay')
        # Player Parameters and rewards
        if True:
            # Uncertainty
            self.reaction_uncertainty = kwargs.get('reaction_uncertainty')
            self.movement_uncertainty = kwargs.get('movement_uncertainty')
            self.timing_uncertainty = kwargs.get('timing_uncertainty')
            self.decision_action_delay_uncertainty = kwargs.get('decision_action_delay_uncertainty')
            self.reaction_plus_movement_uncertainty = np.sqrt(self.reaction_uncertainty**2 + self.movement_uncertainty**2)
            self.total_uncertainty = np.sqrt(self.reaction_plus_movement_uncertainty**2 + self.timing_uncertainty**2)
            self.total_uncertainty_reaction = self.reaction_plus_movement_uncertainty
            self.total_uncertainty_gamble = self.movement_uncertainty 
            self.agent_plus_human_uncertainty = np.sqrt(self.total_uncertainty**2 + self.agent_stds**2)
            # Ability
            self.reaction_time = kwargs.get('reaction_time')
            self.movement_time = kwargs.get('movement_time')
            self.reaction_plus_movement_time = self.reaction_time + self.movement_time
            self.decision_action_delay_mean = kwargs.get('decision_action_delay_mean')
            # Reward and cost values
            self.win_reward = kwargs.get('win_reward',1)
            self.incorrect_cost = kwargs.get('incorrect_cost',0)
            self.indecision_cost = kwargs.get('indecision_cost',0)
            # Prob of selecting the correct target
            self.prob_success_gamble = kwargs.get('prob_success_gamble',0.5)
            self.prob_success_reaction = kwargs.get('prob_success_react',1.0)
    def run_model_fit(self,decision_times):
        self.decision_times = decision_times
        self.decision_time_index = self.decision_times*self.nsteps
        if False:
            # Probabilites
            if True:
                # Prob of selecting reacting or gambling decision
                self.prob_selecting_reaction = self.prob_of_selecting_reaction() # Probability of SELECTING a Decision only depends on timing uncertainty, not total uncertainty
                self.prob_selecting_gamble = 1 - self.prob_selecting_reaction
                
                # Prob of making it to the target
                self.prob_making_reaction = self.prob_making_based_on_agent()
                self.prob_making_gamble = self.prob_reaching_target_gamble()
                
            # Prob of win, incorrect, indecisions
            if True:
                # Probability of receiving a reward (prob_succes multiplied by prob of making it multiplied by probability of actually selecting that action)
                self.prob_win_reaction = self.prob_success_reaction*self.prob_making_reaction*self.prob_selecting_reaction
                self.prob_win_gamble = self.prob_success_gamble*self.prob_making_gamble*self.prob_selecting_gamble
                self.prob_win = self.prob_win_reaction + self.prob_win_gamble
                # Probability of receiving an incorrect cost
                self.prob_incorrect_reaction = (1 - self.prob_success_reaction)*self.prob_making_reaction*self.prob_selecting_reaction
                self.prob_incorrect_gamble = (1 - self.prob_success_gamble)*self.prob_making_gamble*self.prob_selecting_gamble
                self.prob_incorrect = self.prob_incorrect_reaction + self.prob_incorrect_gamble
                # Probability of receiving an indecision cost
                self.prob_indecision_reaction = (1 - self.prob_making_reaction)*self.prob_selecting_reaction
                self.prob_indecision_gamble = (1 - self.prob_making_gamble)*self.prob_selecting_gamble
                self.prob_indecision = self.prob_indecision_reaction + self.prob_indecision_gamble
            # Expected reward 
            if True:
                self.exp_reward_reaction = (self.prob_win_reaction*self.win_reward + self.prob_incorrect_reaction*self.incorrect_cost + self.prob_indecision_reaction*self.indecision_cost)
                self.exp_reward_gamble = (self.prob_win_gamble*self.win_reward + self.prob_incorrect_gamble*self.incorrect_cost + self.prob_indecision_gamble*self.indecision_cost )
                self.exp_reward = self.exp_reward_reaction + self.exp_reward_gamble
                self.exp_reward = self.prob_win*self.win_reward + self.prob_incorrect*self.incorrect_cost + self.prob_indecision*self.indecision_cost 
            
            self.max_exp_reward = np.max(self.exp_reward,axis=1)
            self.metrics_name_dict = {'exp_reward': 'Expected Reward','exp_reward_gamble': 'Expected Reward Gamble','exp_reward_reaction':'Expected Reward Reaction',
                                    'prob_making_reaction': 'Prob Making Reaction','prob_making_gamble':'Prob Making Gamble',
                                    'prob_selecting_reaction':'Prob of Selecting Reaction','prob_selecting_gamble':'Prob of Selecting Gamble',
                                    'prob_win_reaction':'Prob Win Reaction','prob_win_gamble':'Prob Win Gamble',
                                    'prob_incorrect_reaction':'Prob Incorrect Reaction','prob_incorrect_gamble':'Prob Incorrect Gamble',
                                    'prob_indecision_reaction':'Prob Indecision Reaction','prob_indecision_gamble': 'Prob Indecision Gamble',
                                    'prob_win':'Prob Win','prob_incorrect':'Prob Incorrect','prob_indecision':'Prob Indecision',
                                    'prob_making_reaction_based_on_agent':'Prob Making Based on Agent'}
        self.prob_selecting_reaction = self.prob_of_selecting_reaction() # Probability of SELECTING a Decision only depends on timing uncertainty, not total uncertainty
        self.prob_selecting_gamble = 1 - self.prob_selecting_reaction
        
        self.prob_making_reaction = self.prob_making_based_on_agent()
        self.prob_making_gamble = self.prob_making_target_gamble()
        
        # Probability of receiving a reward (prob_succes multiplied by prob of making it multiplied by probability of actually selecting that action)
        self.prob_win_reaction = self.prob_success_reaction*self.prob_making_reaction*self.prob_selecting_reaction
        self.prob_win_gamble = self.prob_success_gamble*self.prob_making_gamble*self.prob_selecting_gamble
        self.prob_win = self.prob_win_reaction + self.prob_win_gamble
        # Probability of receiving an incorrect cost
        self.prob_incorrect_reaction = (1 - self.prob_success_reaction)*self.prob_making_reaction*self.prob_selecting_reaction
        self.prob_incorrect_gamble = (1 - self.prob_success_gamble)*self.prob_making_gamble*self.prob_selecting_gamble
        self.prob_incorrect = self.prob_incorrect_reaction + self.prob_incorrect_gamble
        # Probability of receiving an indecision cost
        self.prob_indecision_reaction = (1 - self.prob_making_reaction)*self.prob_selecting_reaction
        self.prob_indecision_gamble = (1 - self.prob_making_gamble)*self.prob_selecting_gamble
        self.prob_indecision = self.prob_indecision_reaction + self.prob_indecision_gamble
        
        # self.calculate_experiment_metrics_from_expected_reward()
        self.calculate_experiment_metrics()
        
        self.calculate_gamble_reaction_metrics()

        return self.perc_win_calc,self.perc_indecision_calc, self.perc_incorrect_calc
    
    def prob_of_selecting_reaction(self): 
        '''
        This includes the timing uncertainty, and every calculation of win, indecision, incorrect is based 
        on this, so the timing uncertainty is included
        
        The uncertainty here depends on if people have knowledge of their gamble uncertainty or not
            - Gamble uncertainty is larger than peoples timing uncertainty
            - I'm not sure if gamble uncertainty should decide your chances of selecting reaction or not (as of 2/21/23 I'm saying yes it makes a difference)
            
        Instead of comparing every timestep, I'm comparing the decision times from the optimizer to get the prob of selecting reaction and gamble which
        leads to the data
        '''
        output = np.zeros((self.num_blocks))
        for i in range(self.num_blocks):
            if self.known_gamble_uncertainty_on:
                combined_uncertainty = np.sqrt(self.known_gamble_uncertainty**2 + self.agent_stds[i]**2)# + self.decision_action_delay_uncertainty**2) # Prob of SELECTING only includes timing uncertainty and agent uncertainty
            else:
                combined_uncertainty = np.sqrt(self.timing_uncertainty**2 + self.agent_stds[i]**2)# + self.decision_action_delay_uncertainty**2) # Prob of SELECTING only includes timing uncertainty and agent uncertainty
            diff = self.decision_times[i] - self.agent_means[i]
            output[i] = 1 - stats.norm.cdf(0,diff,combined_uncertainty) # I've determined that the decision time just needs to be after, doesn't necessarily need to be after some decision action delay
        return output
    
    def prob_making_based_on_agent(self):
        output = np.zeros((self.num_blocks))
        # Create cut off a as negative infinity
        cut_off_a = np.full_like(self.decision_times,self.neg_inf_cut_off_value)
        # Cut off b is the current timestep
        cut_off_b = self.decision_times
        # self.tiled_agent_means = np.tile(self.agent_means,(2000,1)).T
        # self.tiled_agent_stds = np.tile(self.agent_stds,(2000,1)).T
        self.a, self.b = (cut_off_a - self.agent_means)/self.agent_stds, (cut_off_b - self.agent_means) / self.agent_stds
        self.trunc_agent_mean,self.trunc_agent_var = stats.truncnorm.stats(self.a,self.b,loc=self.agent_means,scale=self.agent_stds) 
        self.trunc_agent_std = np.sqrt(self.trunc_agent_var)
        mean_sum = self.trunc_agent_mean + self.reaction_plus_movement_time
        uncertainty = np.sqrt(self.trunc_agent_std**2 + self.reaction_plus_movement_uncertainty**2)
        output = stats.norm.cdf(1500,mean_sum,uncertainty)
        return output
    
    def prob_making_target_gamble(self):
        if self.known_gamble_delay_on:
            self.gamble_reach_time_mean = self.decision_times + self.movement_time + self.known_gamble_delay # Gamble delay includes decision action delay, but there's another cognitive delay involved
        else:
            self.gamble_reach_time_mean = self.decision_times + self.movement_time + self.decision_action_delay_mean
            
        if self.known_gamble_uncertainty_on:
            self.gamble_reach_time_uncertainty = np.sqrt(self.known_gamble_uncertainty**2 + self.movement_uncertainty**2)
        else:
            self.gamble_reach_time_uncertainty = self.timing_uncertainty # The timing uncertainy measure in coincidence task includes movement uncertainty by nature of the task
        output = stats.norm.cdf(1500,self.gamble_reach_time_mean,self.gamble_reach_time_uncertainty)
        return output
    ##########-------------------------------------###############################
    ##########---- Calculate Fit Metrics ----------###############################
    ##########-------------------------------------###############################
    # Just get the probabilities from the calculations, these are the same as calculating the experiment metrics post-hoc, but those ones
    # can include the unknown shit
    # def calculate_experiment_metrics_from_expected_reward(self):
    #     '''
    #     Get the index of the calculated probabilities
        
    #     This won't be correct if there are unknown uncertainties
    #     '''
    #     self.prob_selecting_reaction_from_known = np.zeros(6)
    #     self.prob_selecting_gamble_from_known = np.zeros(6)
    #     self.prob_making_reaction_from_known = np.zeros(6)
    #     self.prob_making_gamble_from_known = np.zeros(6)
    #     self.prob_indecision_gamble_from_known = np.zeros(6)
    #     self.prob_indecision_reaction_from_known = np.zeros(6)
    #     self.prob_incorrect_gamble_from_known   = np.zeros(6)
    #     self.prob_incorrect_reaction_from_known = np.zeros(6)
    #     self.prob_win_gamble_from_known  = np.zeros(6)
    #     self.prob_win_reaction_from_known = np.zeros(6)
    #     self.prob_win_from_known = np.zeros(6)
    #     self.prob_indecision_from_known = np.zeros(6)
    #     self.prob_incorrect_from_known = np.zeros(6)
    #     for i in range(6):
    #         self.prob_making_reaction_from_known[i] = self.prob_making_reaction[i,self.decision_time_index[i]]
    #         self.prob_making_gamble_from_known[i] = self.prob_making_gamble[i,self.decision_time_index[i]]
    #         self.prob_selecting_reaction_from_known[i] = self.prob_selecting_reaction[i,self.decision_time_index[i]]
    #         self.prob_selecting_gamble_from_known[i] = self.prob_selecting_gamble[i,self.decision_time_index[i]]
    #         self.prob_indecision_gamble_from_known[i] = self.prob_indecision_gamble[i,self.decision_time_index[i]]
    #         self.prob_indecision_reaction_from_known[i] = self.prob_indecision_reaction[i,self.decision_time_index[i]]
    #         self.prob_incorrect_gamble_from_known[i] = self.prob_incorrect_gamble[i,self.decision_time_index[i]]
    #         self.prob_incorrect_reaction_from_known[i] = self.prob_incorrect_reaction[i,self.decision_time_index[i]]
    #         self.prob_win_gamble_from_known[i] = self.prob_win_gamble[i,self.decision_time_index[i]]
    #         self.prob_win_reaction_from_known[i] = self.prob_win_reaction[i,self.decision_time_index[i]]
    #         self.prob_win_from_known[i] = self.prob_win[i,self.decision_time_index[i]]
    #         self.prob_incorrect_from_known[i] = self.prob_incorrect[i,self.decision_time_index[i]]
    #         self.prob_indecision_from_known[i] = self.prob_indecision[i,self.decision_time_index[i]]       
        
    def calculate_gamble_reaction_probs_post(self):
        self.prob_selecting_reaction_fit = np.zeros(6)
        self.prob_selecting_gamble_fit = np.zeros(6)
        # Only reason to recalculate this is if the gamble uncertainty is unknown... if it's known or None, then we go to the else, which can just use the calc during expected reward
        if self.unknown_gamble_uncertainty_on:
            for i in range(6):
                combined_uncertainty = np.sqrt(self.unknown_gamble_uncertainty**2 + self.agent_stds[i]**2)
                diff = self.decision_times[i] - self.agent_means[i]
                self.prob_selecting_reaction_fit[i] = 1 - stats.norm.cdf(0,diff,combined_uncertainty) # I've determined that the decision time just needs to be after, doesn't necessarily need to be after some decision action delay
            self.prob_selecting_gamble_fit = 1 - self.prob_selecting_reaction_fit
        else:
            self.prob_selecting_reaction_fit = self.prob_selecting_reaction
            self.prob_selecting_gamble_fit = self.prob_selecting_gamble
        #Get into percentage 
        self.percent_reactions_fit = self.prob_selecting_reaction_fit*100
        self.percent_gambles_fit = self.prob_selecting_gamble_fit*100
        
    def calculate_mean_leave_target_time(self):
        # Truncated agent mean
        cut_off_a = np.array([self.neg_inf_cut_off_value]*6)
        cut_off_b = np.array(self.decision_times)
        self.a, self.b = (cut_off_a-self.agent_means)/self.agent_stds, (cut_off_b - self.agent_means) / self.agent_stds
        self.trunc_agent_mean_fit,self.trunc_agent_var = stats.truncnorm.stats(self.a,self.b,loc=self.agent_means,scale=self.agent_stds) 
        self.trunc_agent_std_fit = np.sqrt(self.trunc_agent_var)
        
        self.fit_reaction_leave_target_time = self.trunc_agent_mean_fit + self.reaction_time
        
        if self.unknown_gamble_delay_on:
            self.fit_gamble_leave_target_time  = self.decision_times + self.unknown_gamble_delay
        elif self.known_gamble_delay_on:
            self.fit_gamble_leave_target_time  = self.decision_times + self.known_gamble_delay
        else:
            self.fit_gamble_leave_target_time  = self.decision_times + self.decision_action_delay_mean

            
        self.wtd_fit_leave_target_time = (self.prob_selecting_reaction_fit*self.fit_reaction_leave_target_time + \
                                            self.prob_selecting_gamble_fit*self.fit_gamble_leave_target_time)/(self.prob_selecting_gamble_fit+self.prob_selecting_reaction_fit) 
    
    def prob_gamble_indecision_fit(self):
        if self.known_gamble_delay_on:
            self.gamble_reach_time_mean = self.decision_times + self.movement_time + self.known_gamble_delay # Gamble delay includes decision action delay, but there's another cognitive delay involved
        elif self.unknown_gamble_delay_on:
            self.gamble_reach_time_mean = self.decision_times + self.movement_time + self.unknown_gamble_delay
        else:
            self.gamble_reach_time_mean = self.decision_times + self.movement_time + self.decision_action_delay_mean
            
        if self.known_gamble_uncertainty_on:
            self.gamble_reach_time_uncertainty = np.sqrt(self.known_gamble_uncertainty**2 + self.movement_uncertainty**2)
        elif self.unknown_gamble_uncertainty_on:
            self.gamble_reach_time_uncertainty = np.sqrt(self.unknown_gamble_uncertainty**2 + self.movement_uncertainty**2) # The timing uncertainy measure in coincidence task includes
        else:
            self.gamble_reach_time_uncertainty = self.timing_uncertainty

        output = 1 - stats.norm.cdf(1500,self.gamble_reach_time_mean,self.gamble_reach_time_uncertainty)
        return output
    
    def prob_react_indecision_fit(self):
        self.reaction_reach_time_uncertainty = np.sqrt(self.trunc_agent_std_fit**2 + self.reaction_plus_movement_uncertainty**2)
        self.reaction_reach_time_fit = self.trunc_agent_mean_fit + self.reaction_plus_movement_time
        output = 1 - stats.norm.cdf(1500,self.reaction_reach_time_fit,self.reaction_reach_time_uncertainty)
        return output 
    
    def calculate_experiment_metrics(self):
        # Caluclate the probability of reacting and gambling based on the fit decision time
        self.prob_of_selecting_reaction_fit = self.prob_of_selecting_reaction()
        self.calculate_gamble_reaction_probs_post()
        # Calculate the mean leave target time based on the decision time
        self.calculate_mean_leave_target_time()
        self.player_minus_agent_leave_time = self.wtd_fit_leave_target_time - self.agent_means
        
        # Probability of indecision
        self.prob_indecision_if_gamble = self.prob_gamble_indecision_fit()
        self.prob_indecision_if_react = self.prob_react_indecision_fit()        
        
        self.prob_indecision_calc = self.prob_selecting_reaction_fit*self.prob_indecision_if_react + \
                                self.prob_selecting_gamble_fit*self.prob_indecision_if_gamble
        self.perc_indecision_calc = self.prob_indecision_calc*100

        # Probability of winning
        self.prob_win_if_react = (1-self.prob_indecision_if_react)*1.0 # prob win if react is the probability that I don't make an indecision times the probability that i select the right target (1.0)
        self.prob_win_if_gamble = (1-self.prob_indecision_if_gamble)*0.5 # prob win if gamble is the probability that I don't make an indecision times the probabiliyt that I select the right target(0.5)
        self.prob_win_calc = self.prob_selecting_reaction_fit*self.prob_win_if_react + self.prob_selecting_gamble_fit*self.prob_win_if_gamble
        self.perc_win_calc = self.prob_win_calc*100
        
        # Probability of incorrect selection
        self.prob_incorrect_if_react = 0
        self.prob_incorrect_if_gamble = (1-self.prob_indecision_if_gamble)*0.5
        self.prob_incorrect_calc = self.prob_selecting_reaction_fit*self.prob_incorrect_if_react + self.prob_selecting_gamble_fit*self.prob_incorrect_if_gamble
        self.perc_incorrect_calc = self.prob_incorrect_calc*100
  
    def calculate_gamble_reaction_metrics(self):
        # Percent of metric that were reaction and gamble
        self.perc_wins_that_were_reaction = (self.prob_win_if_react*self.prob_selecting_reaction_fit/self.prob_win_calc)*100
        self.perc_wins_that_were_gamble = (self.prob_win_if_gamble*self.prob_selecting_gamble_fit/self.prob_win_calc)*100
        self.perc_incorrects_that_were_reaction = (self.prob_incorrect_if_react*self.prob_selecting_reaction_fit/self.prob_incorrect_calc)*100
        self.perc_incorrects_that_were_gamble = (self.prob_incorrect_if_gamble*self.prob_selecting_gamble_fit/self.prob_incorrect_calc)*100
        self.perc_indecisions_that_were_reaction = (self.prob_indecision_if_react*self.prob_selecting_reaction_fit/self.prob_indecision_calc)*100
        self.perc_indecisions_that_were_gamble = (self.prob_indecision_if_gamble*self.prob_selecting_gamble_fit/self.prob_indecision_calc)*100
        
    def plot_fits(self,metrics,num_plots = None ,dpi=125):
        if num_plots is None:
            num_plots = self.num_blocks
        plt.style.use('cashaback_dark')
        for i in range(num_plots):
            fig,ax = plt.subplots(dpi=dpi)
            for metric in metrics:
                ax.plot(self.timesteps[i,:], getattr(self,metric)[i,:], label = self.metrics_name_dict[metric],zorder=0)
                if metric == 'exp_reward':
                    ax.plot((self.decision_times[i],self.decision_times[i]),(-4,self.max_exp_reward[i]),c='w')
                    ax.scatter(self.decision_times[i],self.max_exp_reward[i],c='w')
                    ax.text(self.decision_times[i],self.max_exp_reward[i]+0.03,f'fit Decision Time = {self.decision_times[i]}',ha = 'center')
            ax.set_ylim(np.min(self.incorrect_cost,self.indecision_cost)-0.03,np.max(self.win_reward)+0.03)
            ax.set_xlim(0,1500)
            ax.set_xticks(np.arange(0,2000,300))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Expected Reward')
            ax.legend(fontsize = 8,loc = (0.01,0.1))
            ax.set_title(f'Gain Function for Decision Time\nAgent Mean,SD = {self.agent_means[i]},{self.agent_stds[i]}')#\n B = {B}')
            plt.show()