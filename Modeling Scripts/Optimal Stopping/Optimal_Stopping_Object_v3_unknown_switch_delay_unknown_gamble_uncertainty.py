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
        # Player Parameters
        self.unknown_switch_delay = kwargs.get('unknown_switch_delay')
        self.unknown_gamble_decision_time_uncertainty = kwargs.get('unknown_gamble_decision_time_uncertainty')
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
       
        # Agent probabilities (not used, agent behavior is used in prob_of_selecting_reaction)
        self.prob_agent_has_gone = self.prob_agent_go()
        self.prob_agent_has_not_gone = 1 - self.prob_agent_has_gone
        
        # Prob of selecting reacting or gambling decision
        self.prob_selecting_reaction = self.prob_of_selecting_reaction() # Probability of SELECTING a Decision only depends on timing uncertainty, not total uncertainty
        self.prob_selecting_gamble = 1 - self.prob_selecting_reaction
        
        # Prob of making it to the target
        self.prob_making_reaction = self.prob_making_based_on_agent()
        self.gamble_reach_time_mean = self.timesteps + self.movement_time + self.decision_action_delay_mean
        self.gamble_reach_time_uncertainty = self.timing_uncertainty
        self.prob_making_gamble = stats.norm.cdf(1500,self.gamble_reach_time_mean,self.gamble_reach_time_uncertainty)
              
        
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

        self.calculate_experiment_metrics_from_expected_reward_prob()
        self.calculate_experiment_metrics()
        
    def prob_agent_go(self):
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        for i in range(self.num_blocks):
            output[i,:] = stats.norm.cdf(self.timesteps[i,:],self.agent_means[i],self.agent_stds[i])
        return output
    
    def prob_of_selecting_reaction(self): 
        '''
        This includes the timing uncertainty, and every calculation of win,indecision,incorrect is based 
        on this, so the timing uncertainty is included
        '''
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        for i in range(self.num_blocks):
            combined_uncertainty = np.sqrt(self.timing_uncertainty**2 + self.agent_stds[i]**2)# + self.decision_action_delay_uncertainty**2) # Prob of SELECTING only includes timing uncertainty and agent uncertainty
            diff = self.timesteps[i,:] - self.agent_means[i]
            # BELOW NEEDS TO INCLUDE THE DECISION ACTION DELAY
            # The probability that selecting at the current timestep will be greater than the agent mean PLUS some decision action delay...
            # So to be able to respond to the agent at 1000ms, I will have needed to held off my decision until 1050 (if decision action delay is 50)

            # Prob that the agent decision time will be LESS than the current timestep
            # Needs to be less than the current timestep minus the decision action delay
            output[i,:] = 1 - stats.norm.cdf(self.decision_action_delay_mean,diff,combined_uncertainty) # Probability that selecting at the current timestep will be greater than the agent mean (aka react)
            output[i,:] = 1 - stats.norm.cdf(0,diff,combined_uncertainty) # Probability that selecting at the current timestep will be greater than the agent mean (aka react)
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
    
    ##########-------------------------------------###############################
    ##########---- Calculate Optimal Metrics ------###############################
    ##########-------------------------------------###############################
    def calculate_mean_leave_target_time(self):
        # Truncated agent mean
        cut_off_a = np.array([self.neg_inf_cut_off_value]*6)
        cut_off_b = np.array(self.optimal_decision_time)
        self.a, self.b = (cut_off_a-self.agent_means)/self.agent_stds, (cut_off_b - self.agent_means) / self.agent_stds
        self.trunc_agent_mean_optimal,self.trunc_agent_var = stats.truncnorm.stats(self.a,self.b,loc=self.agent_means,scale=self.agent_stds) 
        self.trunc_agent_std_optimal = np.sqrt(self.trunc_agent_var)
        self.optimal_reaction_leave_target_time = self.trunc_agent_mean_optimal + self.reaction_time
        self.optimal_gamble_leave_target_time   = self.optimal_decision_time + self.decision_action_delay_mean + self.unknown_switch_delay
            
        self.wtd_optimal_leave_target_time = (self.prob_selecting_reaction_optimal*self.optimal_reaction_leave_target_time + \
                                            self.prob_selecting_gamble_optimal*self.optimal_gamble_leave_target_time)/(self.prob_selecting_gamble_optimal+self.prob_selecting_reaction_optimal) 
    def calculate_gamble_reaction_probs_from_expected_reward(self):
        self.prob_selecting_reaction_optimal = np.zeros(6)
        self.prob_selecting_gamble_optimal = np.zeros(6)
        
        for i in range(6):
            self.prob_selecting_reaction_optimal[i] = self.prob_selecting_reaction[i,self.optimal_index[i]]
            self.prob_selecting_gamble_optimal[i] = self.prob_selecting_gamble[i,self.optimal_index[i]]
        #Get into percentage 
        self.percent_reactions_optimal = self.prob_selecting_reaction_optimal*100
        self.percent_gambles_optimal = self.prob_selecting_gamble_optimal*100
        self.percent_making_reaction_optimal = self.prob_making_reaction_optimal*100
        self.percent_making_gamble_optimal = self.prob_making_gamble_optimal*100
        
    def calculate_gamble_reaction_metrics(self):
        # Percent of metric that were reaction and gamble
        self.perc_wins_that_were_reaction = (self.prob_win_if_react*self.prob_selecting_reaction_optimal/self.prob_win_calc)*100
        self.perc_wins_that_were_gamble = (self.prob_win_if_gamble*self.prob_selecting_gamble_optimal/self.prob_win_calc)*100
        self.perc_incorrects_that_were_reaction = (self.prob_incorrect_if_react*self.prob_selecting_reaction_optimal/self.prob_incorrect_calc)*100
        self.perc_incorrects_that_were_gamble = (self.prob_incorrect_if_gamble*self.prob_selecting_gamble_optimal/self.prob_incorrect_calc)*100
        self.perc_indecisions_that_were_reaction = (self.prob_indecision_if_react*self.prob_selecting_reaction_optimal/self.prob_indecision_calc)*100
        self.perc_indecisions_that_were_gamble = (self.prob_indecision_if_gamble*self.prob_selecting_gamble_optimal/self.prob_indecision_calc)*100

    def calculate_experiment_metrics_from_expected_reward_prob(self):
        '''
        Get the index of the calculated probabilities
        
        I feel like this isn't the way to calculate
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
        
    def calculate_experiment_metrics(self):
        # Caluclate the probability of reacting and gambling based on the optimal decision time
        self.calculate_gamble_reaction_probs_from_expected_reward()
        # Calculate the mean leave target time based on the decision time
        self.calculate_mean_leave_target_time()
        self.prob_indecision_based_on_reach_time = (1 - stats.norm.cdf(1500,self.wtd_optimal_leave_target_time+self.movement_time,self.timing_uncertainty))*100
        # Calculate the expected reach time based on the leave target times
        self.optimal_reach_time_gamble = self.optimal_gamble_leave_target_time + self.movement_time
        self.optimal_reach_time_reaction = self.optimal_reaction_leave_target_time + self.movement_time

        self.player_minus_agent_leave_time = self.wtd_optimal_leave_target_time - self.agent_means
        
        # Prob indecision if I react is
        # 1) Prob that I will react at all N(DT) > N(Agent Decision Time)
        # 2) Prob that my reaction will be under 1500 aka N(ADT) + N(RMT) < 1500
        self.reaction_reach_time_uncertainty = np.sqrt(self.trunc_agent_std_optimal**2 + self.reaction_plus_movement_uncertainty**2)
        self.prob_indecision_if_react = 1 - stats.norm.cdf(1500,self.trunc_agent_mean_optimal + self.reaction_plus_movement_time,self.reaction_reach_time_uncertainty) # Probability that the reach time reaction is > 1500 
        
        self.gamble_reach_time_uncertainty = np.sqrt(self.unknown_gamble_decision_time_uncertainty**2 )
        self.prob_indecision_if_gamble = 1 - stats.norm.cdf(1500,self.optimal_reach_time_gamble,self.gamble_reach_time_uncertainty)
        
        self.prob_indecision_calc = self.prob_selecting_reaction_optimal*self.prob_indecision_if_react + \
                                self.prob_selecting_gamble_optimal*self.prob_indecision_if_gamble
        self.perc_indecision_calc = self.prob_indecision_calc*100

        self.prob_win_if_react = (1-self.prob_indecision_if_react)*1.0 # prob win if react is the probability that I don't make an indecision times the probability that i select the right target (1.0)
        self.prob_win_if_gamble = (1-self.prob_indecision_if_gamble)*0.5 # prob win if gamble is the probability that I don't make an indecision times the probabiliyt that I select the right target(0.5)
        self.prob_win_calc = self.prob_selecting_reaction_optimal*self.prob_win_if_react + self.prob_selecting_gamble_optimal*self.prob_win_if_gamble
        self.perc_win_calc = self.prob_win_calc*100
        
        self.prob_incorrect_if_react = 0
        self.prob_incorrect_if_gamble = (1-self.prob_indecision_if_gamble)*0.5
        self.prob_incorrect_calc = self.prob_selecting_reaction_optimal*self.prob_incorrect_if_react + self.prob_selecting_gamble_optimal*self.prob_incorrect_if_gamble
        self.perc_incorrect_calc = self.prob_incorrect_calc*100

        self.calculate_gamble_reaction_metrics()
        
    def plot_optimals(self,metrics,num_plots = None ,dpi=125):
        if num_plots is None:
            num_plots = self.num_blocks
        plt.style.use('cashaback_dark')
        for i in range(num_plots):
            fig,ax = plt.subplots(dpi=dpi)
            for metric in metrics:
                ax.plot(self.timesteps[i,:], getattr(self,metric)[i,:], label = self.metrics_name_dict[metric],zorder=0)
                if metric == 'exp_reward':
                    ax.plot((self.optimal_decision_time[i],self.optimal_decision_time[i]),(-4,self.max_exp_reward[i]),c='w')
                    ax.scatter(self.optimal_decision_time[i],self.max_exp_reward[i],c='w')
                    ax.text(self.optimal_decision_time[i],self.max_exp_reward[i]+0.03,f'Optimal Decision Time = {self.optimal_decision_time[i]}',ha = 'center')
            ax.set_ylim(np.min(self.incorrect_cost,self.indecision_cost)-0.03,np.max(self.win_reward)+0.03)
            ax.set_xlim(0,1500)
            ax.set_xticks(np.arange(0,2000,300))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Expected Reward')
            ax.legend(fontsize = 8,loc = (0.01,0.1))
            ax.set_title(f'Gain Function for Decision Time\nAgent Mean,SD = {self.agent_means[i]},{self.agent_stds[i]}')#\n B = {B}')
            plt.show()