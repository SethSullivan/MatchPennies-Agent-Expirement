import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import data_visualization as dv
wheel = dv.ColorWheel()

class Optimal_Decision_Time_Model():
    def __init__(self, **kwargs):
        # Task Conditions
        self.num_blocks = kwargs.get('num_blocks',6)
        self.agent_means = kwargs.get('agent_means',np.array([1000,1000,1100,1100,1200,1200]))
        self.agent_stds = kwargs.get('agent_stds',np.array([50,150,50,150,50,150]))
        self.nsteps = 1
        self.timesteps = kwargs.get('timesteps',np.tile(np.arange(0,2000,self.nsteps),(self.num_blocks,1)))
        
        # Player Parameters
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
       
        # Agent probabilities 
        self.prob_agent_has_gone = self.prob_agent_go()
        self.prob_agent_has_not_gone = 1 - self.prob_agent_has_gone
        
        # Prob reacting and gambling
        self.prob_selecting_reaction = self.prob_of_selecting_reaction() # Probability of SELECTING a Decision only depends on timing uncertainty, not total uncertainty
        self.prob_selecting_gamble = 1 - self.prob_selecting_reaction
    
        # Prob of making it on a reaction and gamble 
        self.prob_making_reaction = self.prob_making_when_reacting_or_gambling(react=True)
        self.prob_making_gamble = self.prob_making_when_reacting_or_gambling(gamble=True)
        
        # Probability of receiving a reward (prob_succes multiplied by prob of making it multiplied by probability of actually selecting that action)
        self.prob_reward_reaction = self.prob_success_reaction*self.prob_making_reaction*self.prob_selecting_reaction
        self.prob_reward_gamble = self.prob_success_gamble*self.prob_making_gamble*self.prob_selecting_gamble
        
        # Probability of receiving an incorrect cost
        self.prob_incorrect_cost_reaction = (1 - self.prob_success_reaction)*self.prob_making_reaction*self.prob_selecting_reaction
        self.prob_incorrect_cost_gamble = (1 - self.prob_success_gamble)*self.prob_making_gamble*self.prob_selecting_gamble
        
        # Probability of receiving an indecision cost
        self.prob_indecision_cost_reaction = (1 - self.prob_making_reaction)*self.prob_selecting_reaction
        self.prob_indecision_cost_gamble = (1 - self.prob_making_gamble)*self.prob_selecting_gamble
        
        # Expected reward 
        self.exp_reward_reaction = (self.prob_reward_reaction*self.win_reward + self.prob_incorrect_cost_reaction*self.incorrect_cost + self.prob_indecision_cost_reaction*self.indecision_cost)
        self.exp_reward_gamble = (self.prob_reward_gamble*self.win_reward + self.prob_incorrect_cost_gamble*self.incorrect_cost + self.prob_indecision_cost_gamble*self.indecision_cost )
        self.exp_reward = self.exp_reward_reaction + self.exp_reward_gamble
        
        self.optimal_index = np.argmax(self.exp_reward,axis=1)
        self.optimal_decision_time = np.argmax(self.exp_reward, axis = 1)*self.nsteps
        self.max_exp_reward = np.max(self.exp_reward,axis=1)
        self.metrics_name_dict = {'exp_reward': 'Expected Reward','exp_reward_gamble': 'Expected Reward Gamble','exp_reward_reaction':'Expected Reward Reaction',
                                  'prob_making_reaction': 'Prob Making Reaction','prob_making_gamble':'Prob Making Gamble','prob_agent_has_gone':'Prob Agent Has Gone',
                                  'prob_selecting_reaction':'Prob of Selecting Reaction','prob_reward_reaction':'Prob Reward Reaction','prob_reward_gamble':'Prob Reward Gamble',
                                  'prob_indecision_cost_reaction':'Prob Indecision'}

        self.calculate_experiment_metrics()
        
        
    def prob_agent_go(self):
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        for i in range(self.num_blocks):
            output[i,:] = stats.norm.cdf(self.timesteps[i,:],self.agent_means[i],self.agent_stds[i])
        return output
    
    def prob_of_selecting_reaction(self): 
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        for i in range(self.num_blocks):
            combined_uncertainty = np.sqrt(self.timing_uncertainty**2 + self.agent_stds[i]**2 + self.decision_action_delay_uncertainty**2) # Prob of SELECTING only includes timing uncertainty and agent uncertainty
            diff = self.timesteps[i,:] - self.decision_action_delay_mean - self.agent_means[i]
            # BELOW NEEDS TO INCLUDE THE DECISION ACTION DELAY
            # The probability that selecting at the current timestep will be greater than the agent mean PLUS some decision action delay...
            # So to be able to respond to the agent at 1000ms, I will have needed to held off my decision until 1050 (if decision action delay is 50)

            # Prob that the agent decision time will be LESS than the current timestep
            # Needs to be less than the current timestep minus the decision action delay
            output[i,:] = 1 - stats.norm.cdf(0,diff ,combined_uncertainty) # Probability that selecting at the current timestep will be greater than the agent mean (aka react)
        return output
    
    def prob_making_when_reacting_or_gambling(self, react=False,gamble=False):
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        # Reaction and Gamble decisions have different uncertainties and time to target from decision
        if react:
            uncertainty = self.reaction_plus_movement_uncertainty # No timing uncertainty when reacting
            time_to_target = self.reaction_plus_movement_time 
            for i in range(self.num_blocks):
                # output[i,:] = stats.skewnorm.cdf(1500-self.timesteps[i,:], 2, loc = time_to_target,scale = uncertainty)
                output[i,:] = stats.norm.cdf(1500-self.timesteps[i,:],time_to_target,uncertainty)
        elif gamble:
            uncertainty = np.sqrt(self.movement_uncertainty**2 + self.decision_action_delay_uncertainty**2)# + self.timing_uncertainty**2) # Movement uncertainty and the decision action uncertainty
            time_to_target = self.movement_time + self.decision_action_delay_mean
            for i in range(self.num_blocks):
                output[i,:] = stats.norm.cdf(1500-self.timesteps[i,:],time_to_target,uncertainty)
                
        return output
    
    def calculate_gamble_reaction_probs(self):
        self.prob_selecting_reaction_at_optimal = np.zeros(6)
        self.prob_selecting_gamble_at_optimal = np.zeros(6)
        self.prob_making_reaction_at_optimal = np.zeros(6)
        self.prob_making_gamble_at_optimal = np.zeros(6)
        
        for i in range(6):
            # self.prob_selecting_reaction_at_optimal[i] = self.prob_selecting_reaction[i,self.optimal_index[i]]
            # self.prob_selecting_gamble_at_optimal[i] = self.prob_selecting_gamble[i,self.optimal_index[i]]
            self.prob_making_reaction_at_optimal[i] = self.prob_making_reaction[i,self.optimal_index[i]]
            self.prob_making_gamble_at_optimal[i] = self.prob_making_gamble[i,self.optimal_index[i]]
            
            combined_uncertainty = np.sqrt(self.timing_uncertainty**2 + self.agent_stds[i]**2 + self.decision_action_delay_uncertainty**2) # Prob of SELECTING only includes timing uncertainty and agent uncertainty
            diff = self.optimal_decision_time[i] - self.decision_action_delay_mean - self.agent_means[i] # optimal decision time needs to be 50ms greater than the agent go time for it to be a reaction
            self.prob_selecting_reaction_at_optimal[i] = 1 - stats.norm.cdf(0,diff,combined_uncertainty)
            self.prob_selecting_gamble_at_optimal[i] = 1 - self.prob_selecting_reaction_at_optimal[i]
        #Get into percentage 
        self.percent_reactions_at_optimal = self.prob_selecting_reaction_at_optimal*100
        self.percent_gambles_at_optimal = self.prob_selecting_gamble_at_optimal*100
        self.percent_making_reaction_at_optimal = self.prob_making_reaction_at_optimal*100
        self.percent_making_gamble_at_optimal = self.prob_making_gamble_at_optimal*100
        
    def calculate_mean_leave_target_time(self):
        self.optimal_reaction_leave_target_time = self.optimal_decision_time + self.reaction_time
        self.optimal_gamble_leave_target_time = self.optimal_decision_time + self.decision_action_delay_mean
            
        self.wtd_optimal_leave_target_time = (self.prob_selecting_reaction_at_optimal*self.optimal_reaction_leave_target_time + \
                                            self.prob_selecting_gamble_at_optimal*self.optimal_gamble_leave_target_time)/(self.prob_selecting_gamble_at_optimal+self.prob_selecting_reaction_at_optimal) 
       
        
    def calculate_experiment_metrics(self):
        # Caluclate the probability of reacting and gambling based on the optimal decision time
        self.calculate_gamble_reaction_probs()
        # Calculate the mean leave target time based on the decision time, and probability of that being a reaction or gamble
        self.calculate_mean_leave_target_time()
        
        # Calculate the expected reach time based on the leave target times
        self.optimal_reach_time_gamble = self.optimal_gamble_leave_target_time + self.movement_time
        self.optimal_reach_time_reaction = self.optimal_reaction_leave_target_time + self.movement_time

        self.player_minus_agent_leave_time = self.wtd_optimal_leave_target_time - self.agent_means
        # # Calculate the probability of selecting a reaction decision or gamble decision based on timing uncertainty and decision time
        # mean_diff = self.optimal_decision_time  - (self.agent_means + self.decision_action_delay_mean)
        # std_diff = np.sqrt(self.timing_uncertainty**2 + self.agent_stds**2) 
        # self.prob_selecting_reaction_at_optimal = 1 - stats.norm.cdf(0, mean_diff, std_diff) # Probability that optimal decision time is greater than the agent decision time plus some delay after the agent goes (aka we react) 
        # self.prob_selecting_gamble_at_optimal = 1 - self.prob_selecting_reaction_at_optimal   
        # prob_indecision = (1 - stats.norm.cdf(1500-group.combine_all_subjects('player_task_decision_time_mean'),100,group.combine_all_subjects('gamble_decision_time_sd')))*100
        
        # Prob indecision if I react is
        # 1) Prob that I will react at all N(DT) > N(Agent Decision Time)
        # 2) Prob that my reaction will be under 1500 aka N(ADT) + N(RMT) < 1500
        self.reaction_reach_time_uncertainty = np.sqrt(self.agent_stds**2 + self.reaction_plus_movement_uncertainty**2)
        
        #Optimal reach time reactino includes the truncated agent means
        self.prob_indecision_if_react = 1 - stats.norm.cdf(1500,self.optimal_reach_time_reaction,self.reaction_reach_time_uncertainty) # Probability that the reach time reaction is > 1500 
        
        self.gamble_reach_time_uncertainty = np.sqrt(self.timing_uncertainty**2 + self.movement_uncertainty**2)
        self.prob_indecision_if_gamble = 1 - stats.norm.cdf(1500,self.optimal_reach_time_gamble,self.gamble_reach_time_uncertainty)
        
        self.prob_indecision = self.prob_selecting_reaction_at_optimal*self.prob_indecision_if_react + \
                                self.prob_selecting_gamble_at_optimal*self.prob_indecision_if_gamble
        self.perc_indecision = self.prob_indecision*100

        self.prob_win_if_react = (1-self.prob_indecision_if_react)*1.0 # prob win if react is the probability that I don't make an indecision times the probability that i select the right target (1.0)
        self.prob_win_if_gamble = (1-self.prob_indecision_if_gamble)*0.5 # prob win if gamble is the probability that I don't make an indecision times the probabiliyt that I select the right target(0.5)
        self.prob_win = self.prob_selecting_reaction_at_optimal*self.prob_win_if_react + self.prob_selecting_gamble_at_optimal*self.prob_win_if_gamble
        self.perc_win = self.prob_win*100
        
        self.prob_incorrect_if_react = 0
        self.prob_incorrect_if_gamble = (1-self.prob_indecision_if_gamble)*0.5
        self.prob_incorrect = self.prob_selecting_reaction_at_optimal*self.prob_incorrect_if_react + self.prob_selecting_gamble_at_optimal*self.prob_incorrect_if_gamble
        self.perc_incorrect = self.prob_incorrect*100

    def plot_optimals(self,metrics,dpi=125):
        plt.style.use('cashaback_dark')
        for i in range(self.num_blocks):
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