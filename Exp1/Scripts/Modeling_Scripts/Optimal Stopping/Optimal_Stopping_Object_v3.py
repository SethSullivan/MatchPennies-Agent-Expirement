import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import data_visualization as dv
wheel = dv.ColorWheel()
'''
1/19/23 - Realized that the multiplication of two PDFS is NOT the same as the convolution of those PDFS


- Where the convolution comes in is when you're adding two distributions or subtracting
- But the convolution of them is the same as adding means and adding variances and taking cdf or pdf (at least with normal distributions)

- ***This script now makes it so that the probability of making an indecision depends on the agent and my reaction plus movement time and 
uncertainty. The convolution of these at every time point is what I want***

1/20/23
- Now using truncated mean and std
    - Used in the calculation of the leave time
- Need to iron out exactly what the indecision calculations are
- The INDECISIONS ARE NOW

1/31/23 

I've realized I wasn't adding the trunc mean in the trunc mean function to the reaction time
- Once I implemented this, the model makes perfect sense
- This is the clean version of Optimal_Stopping_Object_v2_wtd_reach_time_each_timestep_v2
     - Except I don't think I'm weighing it at each time step
     - Still need to figure out how to multiply prob_selecting_reaction and prob_selecting_gamble
            - When do I multiply them when considering the reach time???
            
- This script should incorporate the weighted reach time as well as the trunc agent + reaction time for the
  reaction reach time... 
  
  
2/15/23 - Prob of making based on gamble
- Each timestep is the INTENDED decision time, tau
- So the actual decision time is around that distribution
- THEREFORE, the probability of making it on a gamble depends on the timing uncertainty as well
- So it's the timing distribution around the current timestep plus the movement distribution 
- We do this for EVERY TIMESTEP


'''
class Optimal_Decision_Time_Model():
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
            self.weird_reaction_gamble_cutoff = kwargs.get('weird_reaction_gamble_cutoff',0)

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
    def run_model(self):
        # Probabilites
        if True:
            # Agent probabilities (not used, agent behavior is used in prob_of_selecting_reaction)
            self.prob_agent_has_gone = self.prob_agent_go()
            self.prob_agent_has_not_gone = 1 - self.prob_agent_has_gone
            
            # Prob of selecting reacting or gambling decision
            self.prob_selecting_reaction = self.prob_of_selecting_reaction() # Probability of SELECTING a Decision only depends on timing uncertainty, not total uncertainty
            self.prob_selecting_gamble = 1 - self.prob_selecting_reaction
            
            # Prob of making it to the target
            self.prob_making_reaction = self.prob_making_based_on_agent()
            self.prob_making_gamble = self.prob_making_for_gamble()
              
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
        
        self.optimal_index = np.argmax(self.exp_reward,axis=1)
        self.optimal_decision_time = np.argmax(self.exp_reward, axis = 1)*self.nsteps
        self.max_exp_reward = np.max(self.exp_reward,axis=1)
        self.metrics_name_dict = {'exp_reward': 'Expected Reward','exp_reward_gamble': 'Expected Reward Gamble','exp_reward_reaction':'Expected Reward Reaction',
                                  'prob_making_reaction': 'Prob Making Reaction','prob_making_gamble':'Prob Making Gamble','prob_agent_has_gone':'Prob Agent Has Gone',
                                  'prob_selecting_reaction':'Prob of Selecting Reaction','prob_selecting_gamble':'Prob of Selecting Gamble',
                                  'prob_win_reaction':'Prob Win Reaction','prob_win_gamble':'Prob Win Gamble',
                                  'prob_incorrect_reaction':'Prob Incorrect Reaction','prob_incorrect_gamble':'Prob Incorrect Gamble',
                                  'prob_indecision_reaction':'Prob Indecision Reaction','prob_indecision_gamble': 'Prob Indecision Gamble',
                                  'prob_win':'Prob Win','prob_incorrect':'Prob Incorrect','prob_indecision':'Prob Indecision',
                                  'prob_making_reaction_based_on_agent':'Prob Making Based on Agent'}

        self.calculate_experiment_metrics_from_expected_reward()
        self.calculate_experiment_metrics()
        
        self.calculate_gamble_reaction_metrics()

        
    def prob_agent_go(self):
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        for i in range(self.num_blocks):
            output[i,:] = stats.norm.cdf(self.timesteps[i,:],self.agent_means[i],self.agent_stds[i])
        return output
    
    def prob_of_selecting_reaction(self): 
        '''
        This includes the timing uncertainty, and every calculation of win, indecision, incorrect is based 
        on this, so the timing uncertainty is included
        
        The uncertainty here depends on if people have knowledge of their gamble uncertainty or not
            - Gamble uncertainty is larger than peoples timing uncertainty
            - I'm not sure if gamble uncertainty should decide your chances of selecting reaction or not (as of 2/21/23 I'm saying yes it makes a difference)
        '''
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        if self.known_gamble_uncertainty_on:
            combined_uncertainty = np.sqrt(self.known_gamble_uncertainty**2 + self.agent_stds**2)# + self.decision_action_delay_uncertainty**2) # Prob of SELECTING only includes timing uncertainty and agent uncertainty
            combined_uncertainty = np.tile(combined_uncertainty,(2000,1)).T
        else:
            combined_uncertainty = np.sqrt(self.timing_uncertainty**2 + self.agent_stds**2)# + self.decision_action_delay_uncertainty**2) # Prob of SELECTING only includes timing uncertainty and agent uncertainty
        
        for i in range(self.num_blocks):   
            diff = self.timesteps[i,:] - self.agent_means[i]
            if self.known_gamble_delay_on:
                output[i,:] = 1 - stats.norm.cdf(self.weird_reaction_gamble_cutoff,diff,combined_uncertainty[i]) # I've determined that the decision time just needs to be after, doesn't necessarily need to be after some decision action delay
            elif self.unknown_gamble_delay_on:
                output[i,:] = 1 - stats.norm.cdf(self.weird_reaction_gamble_cutoff,diff,combined_uncertainty[i]) # I've determined that the decision time just needs to be after, doesn't necessarily need to be after some decision action delay
            else:
                output[i,:] = 1 - stats.norm.cdf(self.weird_reaction_gamble_cutoff,diff,combined_uncertainty[i]) # I've determined that the decision time just needs to be after, doesn't necessarily need to be after some decision action delay
                
        return output
    
    def prob_making_based_on_agent(self):
        output = np.zeros((self.num_blocks))
        # Create cut off a as negative infinity
        cut_off_a = np.full_like(self.timesteps,self.neg_inf_cut_off_value)
        # Cut off b is the current timestep
        cut_off_b = self.timesteps
        self.tiled_agent_means = np.tile(self.agent_means,(2000,1)).T
        self.tiled_agent_stds = np.tile(self.agent_stds,(2000,1)).T
        self.a, self.b = (cut_off_a - self.tiled_agent_means)/self.tiled_agent_stds, (cut_off_b - self.tiled_agent_means) / self.tiled_agent_stds
        self.trunc_agent_mean_every_timestep,self.trunc_agent_var_every_timestep = stats.truncnorm.stats(self.a,self.b,loc=self.tiled_agent_means,scale=self.tiled_agent_stds) 
        self.trunc_agent_std_every_timestep = np.sqrt(self.trunc_agent_var_every_timestep)
        mean_sum = self.trunc_agent_mean_every_timestep + self.reaction_plus_movement_time
        uncertainty = np.sqrt(self.trunc_agent_std_every_timestep**2 + self.reaction_plus_movement_uncertainty**2)
        output = stats.norm.cdf(1500,mean_sum,uncertainty)
        return output
    
    def prob_making_for_gamble(self):
        if self.known_gamble_delay_on:
            self.gamble_reach_time_mean = self.timesteps + self.movement_time + self.known_gamble_delay # Gamble delay includes decision action delay, but there's another cognitive delay involved
        else:
            self.gamble_reach_time_mean = self.timesteps + self.movement_time + self.decision_action_delay_mean
        if self.known_gamble_uncertainty_on:
            self.gamble_reach_time_uncertainty = np.sqrt(self.known_gamble_uncertainty**2 + self.movement_uncertainty**2)
        else:
            self.gamble_reach_time_uncertainty = self.timing_uncertainty # The timing uncertainy measure in coincidence task includes 
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        for i in range(self.num_blocks): 
            output[i,:] = stats.norm.cdf(1500,self.gamble_reach_time_mean[i,:],self.gamble_reach_time_uncertainty[i])
        return output
    ##########-------------------------------------###############################
    ##########---- Calculate Optimal Metrics ------###############################
    ##########-------------------------------------###############################
    
    # Just get the probabilities from the calculations, these are the same as calculating the experiment metrics post-hoc, but those ones
    # can include the unknown shit
    def calculate_experiment_metrics_from_expected_reward(self):
        '''
        Get the index of the calculated probabilities
        
        This won't be correct if there are unknown uncertainties
        '''
        self.prob_selecting_reaction_optimal = np.zeros(6)
        self.prob_selecting_gamble_optimal = np.zeros(6)
        self.prob_making_reaction_optimal = np.zeros(6)
        self.prob_making_gamble_optimal = np.zeros(6)
        self.prob_indecision_gamble_optimal = np.zeros(6)
        self.prob_indecision_reaction_optimal = np.zeros(6)
        self.prob_incorrect_gamble_optimal   = np.zeros(6)
        self.prob_incorrect_reaction_optimal = np.zeros(6)
        self.prob_win_gamble_optimal  = np.zeros(6)
        self.prob_win_reaction_optimal = np.zeros(6)
        self.prob_win_optimal = np.zeros(6)
        self.prob_indecision_optimal = np.zeros(6)
        self.prob_incorrect_optimal = np.zeros(6)
        for i in range(6):
            self.prob_making_reaction_optimal[i] = self.prob_making_reaction[i,self.optimal_index[i]]
            self.prob_making_gamble_optimal[i] = self.prob_making_gamble[i,self.optimal_index[i]]
            self.prob_selecting_reaction_optimal[i] = self.prob_selecting_reaction[i,self.optimal_index[i]]
            self.prob_selecting_gamble_optimal[i] = self.prob_selecting_gamble[i,self.optimal_index[i]]
            self.prob_indecision_gamble_optimal[i] = self.prob_indecision_gamble[i,self.optimal_index[i]]
            self.prob_indecision_reaction_optimal[i] = self.prob_indecision_reaction[i,self.optimal_index[i]]
            self.prob_incorrect_gamble_optimal[i] = self.prob_incorrect_gamble[i,self.optimal_index[i]]
            self.prob_incorrect_reaction_optimal[i] = self.prob_incorrect_reaction[i,self.optimal_index[i]]
            self.prob_win_gamble_optimal[i] = self.prob_win_gamble[i,self.optimal_index[i]]
            self.prob_win_reaction_optimal[i] = self.prob_win_reaction[i,self.optimal_index[i]]
            self.prob_win_optimal[i] = self.prob_win[i,self.optimal_index[i]]
            self.prob_incorrect_optimal[i] = self.prob_incorrect[i,self.optimal_index[i]]
            self.prob_indecision_optimal[i] = self.prob_indecision[i,self.optimal_index[i]]
            
    def calculate_gamble_reaction_probs_from_expected_reward(self):
        self.prob_selecting_reaction_optimal = np.zeros(6)
        self.prob_selecting_gamble_optimal = np.zeros(6)
        # Only reason to recalculate this is if the gamble uncertainty is unknown... if it's known or None, then we go to the else, which can just use the calc during expected reward
        if self.unknown_gamble_uncertainty_on:
            for i in range(6):
                combined_uncertainty = np.sqrt(self.unknown_gamble_uncertainty**2 + self.agent_stds[i]**2)
                diff = self.optimal_index[i] - self.agent_means[i]
                self.prob_selecting_reaction_optimal[i] = 1 - stats.norm.cdf(self.weird_reaction_gamble_cutoff,diff,combined_uncertainty) # I've determined that the decision time just needs to be after, doesn't necessarily need to be after some decision action delay
            self.prob_selecting_gamble_optimal = 1 - self.prob_selecting_reaction_optimal
        else:
            for i in range(6):
                self.prob_selecting_reaction_optimal[i] = self.prob_selecting_reaction[i,self.optimal_index[i]]
                self.prob_selecting_gamble_optimal[i] = self.prob_selecting_gamble[i,self.optimal_index[i]]
        #Get into percentage 
        self.percent_reactions_optimal = self.prob_selecting_reaction_optimal*100
        self.percent_gambles_optimal = self.prob_selecting_gamble_optimal*100
        self.percent_making_reaction_optimal = self.prob_making_reaction_optimal*100
        self.percent_making_gamble_optimal = self.prob_making_gamble_optimal*100
        
    def calculate_mean_leave_target_time(self):
        # Truncated agent mean
        cut_off_a = np.array([self.neg_inf_cut_off_value]*6)
        cut_off_b = np.array(self.optimal_decision_time)
        self.a, self.b = (cut_off_a-self.agent_means)/self.agent_stds, (cut_off_b - self.agent_means) / self.agent_stds
        self.trunc_agent_mean_optimal,self.trunc_agent_var = stats.truncnorm.stats(self.a,self.b,loc=self.agent_means,scale=self.agent_stds) 
        self.trunc_agent_std_optimal = np.sqrt(self.trunc_agent_var)
        
        self.optimal_reaction_leave_target_time = self.trunc_agent_mean_optimal + self.reaction_time
        self.optimal_reaction_leave_target_time_sd = np.sqrt(self.trunc_agent_std_optimal**2 + self.reaction_uncertainty**2)
        if self.unknown_gamble_delay_on:
            self.optimal_gamble_leave_target_time   = self.optimal_decision_time + self.unknown_gamble_delay
        elif self.known_gamble_delay_on:
            self.optimal_gamble_leave_target_time   = self.optimal_decision_time + self.known_gamble_delay
        else:
            self.optimal_gamble_leave_target_time   = self.optimal_decision_time + self.decision_action_delay_mean
        
        if self.unknown_gamble_uncertainty_on:
            self.optimal_gamble_leave_target_time_sd   = np.sqrt(self.trunc_agent_std_optimal**2 + self.unknown_gamble_uncertainty**2)
        elif self.known_gamble_uncertainty_on:
            self.optimal_gamble_leave_target_time_sd   = np.sqrt(self.trunc_agent_std_optimal**2 + self.known_gamble_uncertainty**2)
        else:
            self.optimal_gamble_leave_target_time_sd   = np.sqrt(self.trunc_agent_std_optimal**2 + self.timing_uncertainty**2)
        
            
        self.wtd_optimal_leave_target_time = (self.prob_selecting_reaction_optimal*self.optimal_reaction_leave_target_time + \
                                            self.prob_selecting_gamble_optimal*self.optimal_gamble_leave_target_time)/(self.prob_selecting_gamble_optimal+self.prob_selecting_reaction_optimal) 
        self.wtd_optimal_reach_target_time = self.wtd_optimal_leave_target_time + self.movement_time
    
    def prob_gamble_indecision_optimal(self):
        if self.known_gamble_delay_on:
            self.gamble_reach_time_mean = self.optimal_decision_time + self.movement_time + self.known_gamble_delay # Gamble delay includes decision action delay, but there's another cognitive delay involved
        elif self.unknown_gamble_delay_on:
            self.gamble_reach_time_mean = self.optimal_decision_time + self.movement_time + self.unknown_gamble_delay
        else:
            self.gamble_reach_time_mean = self.optimal_decision_time + self.movement_time + self.decision_action_delay_mean
            
        if self.known_gamble_uncertainty_on:
            self.gamble_reach_time_uncertainty = np.sqrt(self.known_gamble_uncertainty**2 + self.movement_uncertainty**2)
        elif self.unknown_gamble_uncertainty_on:
            self.gamble_reach_time_uncertainty = np.sqrt(self.unknown_gamble_uncertainty**2 + self.movement_uncertainty**2) # The timing uncertainy measure in coincidence task includes
        else:
            self.gamble_reach_time_uncertainty = self.timing_uncertainty

        output = 1 - stats.norm.cdf(1500,self.gamble_reach_time_mean,self.gamble_reach_time_uncertainty)
        return output
    
    def prob_react_indecision_optimal(self):
        self.reaction_reach_time_uncertainty = np.sqrt(self.trunc_agent_std_optimal**2 + self.reaction_plus_movement_uncertainty**2)
        self.reaction_reach_time_optimal = self.trunc_agent_mean_optimal + self.reaction_plus_movement_time
        output = 1 - stats.norm.cdf(1500,self.reaction_reach_time_optimal,self.reaction_reach_time_uncertainty)
        return output 
    
    def calculate_experiment_metrics(self):
        # Caluclate the probability of reacting and gambling based on the optimal decision time
        self.calculate_gamble_reaction_probs_from_expected_reward()
        # Calculate the mean leave target time based on the decision time
        self.calculate_mean_leave_target_time()
        self.player_minus_agent_leave_time = self.wtd_optimal_leave_target_time - self.agent_means
        self.player_minus_agent_reaction_leave_time = self.reaction_time
        self.player_minus_agent_gamble_leave_time = self.optimal_gamble_leave_target_time - self.agent_means
        # Probability of indecision
        self.prob_indecision_if_gamble = self.prob_gamble_indecision_optimal()
        self.prob_indecision_if_react = self.prob_react_indecision_optimal()        
        
        self.prob_indecision_calc = self.prob_selecting_reaction_optimal*self.prob_indecision_if_react + \
                                self.prob_selecting_gamble_optimal*self.prob_indecision_if_gamble
        self.perc_indecision_calc = self.prob_indecision_calc*100

        # Probability of winning
        self.prob_win_if_react = (1-self.prob_indecision_if_react)*1.0 # prob win if react is the probability that I don't make an indecision times the probability that i select the right target (1.0)
        self.prob_win_if_gamble = (1-self.prob_indecision_if_gamble)*0.5 # prob win if gamble is the probability that I don't make an indecision times the probabiliyt that I select the right target(0.5)
        self.prob_win_calc = self.prob_selecting_reaction_optimal*self.prob_win_if_react + self.prob_selecting_gamble_optimal*self.prob_win_if_gamble
        self.perc_win_calc = self.prob_win_calc*100
        
        # Probability of incorrect selection
        self.prob_incorrect_if_react = 0
        self.prob_incorrect_if_gamble = (1-self.prob_indecision_if_gamble)*0.5
        self.prob_incorrect_calc = self.prob_selecting_reaction_optimal*self.prob_incorrect_if_react + self.prob_selecting_gamble_optimal*self.prob_incorrect_if_gamble
        
        self.perc_incorrect_calc = self.prob_incorrect_calc*100
  
    def calculate_gamble_reaction_metrics(self):
        self.temp_prob_win = self.replace_zero_with_nan(self.prob_win_calc)
        self.temp_prob_indecision = self.replace_zero_with_nan(self.prob_indecision_calc)
        self.temp_prob_incorrect = self.replace_zero_with_nan(self.prob_incorrect_calc)
        # Percent of metric that were reaction and gamble
        self.perc_wins_that_were_gamble          = ((self.prob_win_if_gamble*self.prob_selecting_gamble_optimal)/self.temp_prob_win)*100
        self.perc_indecisions_that_were_gamble   = ((self.prob_indecision_if_gamble*self.prob_selecting_gamble_optimal)/self.temp_prob_indecision)*100
        self.perc_incorrects_that_were_gamble    = ((self.prob_incorrect_if_gamble*self.prob_selecting_gamble_optimal)/self.temp_prob_incorrect)*100
        self.perc_wins_that_were_reaction        = ((self.prob_win_if_react*self.prob_selecting_reaction_optimal)/self.temp_prob_win)*100
        self.perc_indecisions_that_were_reaction = ((self.prob_indecision_if_react*self.prob_selecting_reaction_optimal)/self.temp_prob_indecision)*100
        self.perc_incorrects_that_were_reaction  = ((self.prob_incorrect_if_react*self.prob_selecting_reaction_optimal)/self.temp_prob_incorrect)*100
        
        self.perc_gambles_that_were_wins          = ((self.prob_win_if_gamble*self.prob_selecting_gamble_optimal)/self.prob_selecting_gamble_optimal)*100
        self.perc_gambles_that_were_incorrects    = ((self.prob_incorrect_if_gamble*self.prob_selecting_gamble_optimal)/self.prob_selecting_gamble_optimal)*100
        self.perc_gambles_that_were_indecisions   = ((self.prob_indecision_if_gamble*self.prob_selecting_gamble_optimal)/self.prob_selecting_gamble_optimal)*100
        self.perc_reactions_that_were_wins        = ((self.prob_win_if_react*self.prob_selecting_reaction_optimal)/self.prob_selecting_reaction_optimal)*100
        self.perc_reactions_that_were_incorrects  = ((self.prob_incorrect_if_react*self.prob_selecting_reaction_optimal)/self.prob_selecting_reaction_optimal)*100
        self.perc_reactions_that_were_indecisions = ((self.prob_indecision_if_react*self.prob_selecting_reaction_optimal)/self.prob_selecting_reaction_optimal)*100
    
    
    def mseloss(self,decision_time):
        # Go through the model with these specific DECISION times
        self.calculate_metrics_with_certain_decision_time(decision_time)
        
        # Get wins,indecisions,incorrects,and leave target times and compare to data
        win_diff = abs(self.perc_win_calc - self.tune_data[0])
        indecision_diff = abs(self.perc_indecision_calc - self.tune_data[1])
        incorrect_diff = abs(self.perc_incorrect_calc - self.tune_data[2])
        leave_target_time_diff = abs(self.wtd_optimal_leave_target_time - self.tune_data[3])
        perc_reactions_diff = abs(self.prob_selecting_reaction_optimal*100 - self.tune_data[4])
        perc_gambles_diff = abs(self.prob_selecting_gamble_optimal*100 - self.tune_data[5])
        reaction_leave_time_diff = abs(self.optimal_reaction_leave_target_time - self.tune_data[6])
        gamble_leave_time_diff = abs(self.optimal_gamble_leave_target_time - self.tune_data[7])
        
        metric_loss = np.array([win_diff,indecision_diff,incorrect_diff,leave_target_time_diff,
                                perc_reactions_diff,perc_gambles_diff,reaction_leave_time_diff,gamble_leave_time_diff])
        return metric_loss
    
    def fit_model_to_data(self,data):
        '''
        data = [wins,indecisions,incorrects,decision_times,perc_reaction_decisions,perc_gamble_decisions]
        '''
        self.tune_data = data
        self.tune_timesteps = np.arange(900,1800,1)
        decision_times = np.array([self.tune_timesteps[0]]*6) # Start off with 600 for each parameter
        num_metrics= len(self.tune_data)
        loss_store = np.zeros((num_metrics,6,len(self.tune_timesteps))) # Each metric,each block, each timestep
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
        
    def calculate_metrics_with_certain_decision_time(self,decision_times):
        self.optimal_decision_time = decision_times
        self.optimal_index = (decision_times/self.nsteps).astype(int)

        self.calculate_experiment_metrics_from_expected_reward()
        self.calculate_experiment_metrics()
        self.calculate_gamble_reaction_metrics()
    
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
                ax.plot(self.timesteps[i,:], getattr(self,metric)[i,:], label = self.metrics_name_dict[metric],zorder=0)
                if metric == 'exp_reward':
                    ax.plot((self.optimal_decision_time[i],self.optimal_decision_time[i]),(-4,self.exp_reward[i,self.optimal_index[i]]),c='w')
                    ax.scatter(self.optimal_decision_time[i],self.exp_reward[i,self.optimal_index[i]],c='w')
                    ax.text(self.optimal_decision_time[i],self.exp_reward[i,self.optimal_index[i]]+0.03,f'Optimal Decision Time = {self.optimal_decision_time[i]}',ha = 'center')
            ax.set_ylim(np.min(self.incorrect_cost,self.indecision_cost)-0.03,np.max(self.win_reward)+0.03)
            ax.set_xlim(0,1500)
            ax.set_xticks(np.arange(0,2000,300))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Expected Reward')
            ax.legend(fontsize = 8,loc = (0.01,0.1))
            ax.set_title(f'Gain Function for Decision Time\nAgent Mean,SD = {self.agent_means[i]},{self.agent_stds[i]}')#\n B = {B}')
            plt.show()