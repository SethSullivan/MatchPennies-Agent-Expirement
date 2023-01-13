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
        self.prob_selecting_reaction = self.prob_of_selecting_reaction()
        self.prob_selecting_gamble = 1 - self.prob_selecting_reaction
    
        # Prob of making it on a reaction and gamble 
        self.prob_making_reaction = self.prob_making_when_reacting_or_gambling(react=True)
        self.prob_making_gamble = self.prob_making_when_reacting_or_gambling(gamble=True)
        
        # Probability of receiving a reward (prob_succes multiplied by prob of making it multiplied by probability of being able to select that action)
        self.prob_reward_reaction = self.prob_success_reaction*self.prob_making_reaction*self.prob_selecting_reaction
        self.prob_reward_gamble = self.prob_success_gamble*self.prob_making_gamble*self.prob_selecting_gamble
        
        # Probability of receiving an incorrect cost
        self.prob_incorrect_cost_reaction = (1 - self.prob_success_reaction)*self.prob_making_reaction*self.prob_selecting_reaction
        self.prob_incorrect_cost_gamble = (1 - self.prob_success_gamble)*self.prob_making_gamble*self.prob_selecting_gamble
        
        # Probability of receiving an indecision cost
        self.prob_indecision_cost_reaction = (1 - self.prob_making_reaction)*self.prob_selecting_reaction
        self.prob_indecision_cost_gamble = (1 - self.prob_making_gamble)*self.prob_selecting_gamble
        
        # Expected reward 
        self.exp_reward_reaction = self.prob_reward_reaction*self.win_reward + self.prob_incorrect_cost_reaction*self.incorrect_cost + self.prob_indecision_cost_reaction*self.indecision_cost 
        self.exp_reward_gamble = self.prob_reward_gamble*self.win_reward + self.prob_incorrect_cost_gamble*self.incorrect_cost + self.prob_indecision_cost_gamble*self.indecision_cost 
        self.exp_reward = self.exp_reward_reaction + self.exp_reward_gamble
        
        self.optimal_index = np.argmax(self.exp_reward,axis=1)
        self.optimal_decision_time = np.argmax(self.exp_reward, axis = 1)*self.nsteps
        self.max_exp_reward = np.max(self.exp_reward,axis=1)
        self.metrics_name_dict = {'exp_reward': 'Expected Reward','exp_reward_gamble': 'Expected Reward Gamble','exp_reward_reaction':'Expected Reward Reaction',
                                  'prob_making_reaction': 'Prob Making Reaction','prob_making_gamble':'Prob Making Gamble','prob_agent_has_gone':'Prob Agent Has Gone',
                                  'prob_selecting_reaction':'Prob of Selecting Reaction'}
        self.calculate_mean_leave_target_time()
    def prob_agent_go(self):
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        for i in range(self.num_blocks):
            output[i,:] = stats.norm.cdf(self.timesteps[i,:],self.agent_means[i],self.agent_stds[i])
        return output
    
    def prob_of_selecting_reaction(self):
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        for i in range(self.num_blocks):
            combined_uncertainty = np.sqrt(self.total_uncertainty**2 + self.agent_stds[i]**2)
            output[i,:] = stats.norm.cdf(self.timesteps[i,:] - self.agent_means[i],0,combined_uncertainty) # Probability that selecting at the current timestep will be greater than the agent mean (aka react)
        return output
    
    def prob_making_when_reacting_or_gambling(self, react=False,gamble=False):
        output = np.zeros((self.num_blocks,len(self.timesteps[0,:])))
        # Reaction and Gamble decisions have different uncertainties and time to target from decision
        if react:
            uncertainty = self.reaction_plus_movement_uncertainty
            time_to_target = self.reaction_plus_movement_time 
        elif gamble:
            uncertainty = np.sqrt(self.movement_uncertainty**2 + self.decision_action_delay_uncertainty**2)
            time_to_target = self.movement_time + self.decision_action_delay_mean
        for i in range(self.num_blocks):
            output[i,:] = stats.norm.cdf(1500-self.timesteps[i,:],time_to_target,uncertainty)
        return output
    def calculate_mean_leave_target_time(self):
        self.optimal_reaction_leave_target_time = self.optimal_decision_time + self.reaction_time
        self.optimal_gamble_leave_target_time = self.optimal_decision_time + self.decision_action_delay_mean
        self.prob_selecting_reaction_at_optimal = []
        self.prob_selecting_gamble_at_optimal = []
        for i in range(6):
            self.prob_selecting_reaction_at_optimal.append(self.prob_selecting_reaction[i,self.optimal_index[i]])
            self.prob_selecting_gamble_at_optimal.append(self.prob_selecting_gamble[i,self.optimal_index[i]])
        self.wtd_optimal_leave_target_time = self.prob_selecting_reaction_at_optimal*self.optimal_reaction_leave_target_time + \
                                            self.prob_selecting_gamble_at_optimal*self.optimal_gamble_leave_target_time 
    
    def plot_optimals(self,metrics,dpi=125):
        plt.style.use('cashaback_dark')
        for i in range(self.num_blocks):
            fig,ax = plt.subplots(dpi=dpi)
            for metric in metrics:
                ax.plot(self.timesteps[i,:], getattr(self,metric)[i,:], label = self.metrics_name_dict[metric],zorder=0)
                if metric == 'exp_reward':
                    ax.plot((self.optimal_decision_time[i],self.optimal_decision_time[i]),(0,self.max_exp_reward[i]),c='w')
                    ax.scatter(self.optimal_decision_time[i],self.max_exp_reward[i],c='w')
                    ax.text(self.optimal_decision_time[i],self.max_exp_reward[i]+0.03,f'Optimal Decision Time = {self.optimal_decision_time[i]}',horizontalalignment = 'center')
            ax.set_ylim(np.min(self.incorrect_cost,self.indecision_cost)-0.03,np.max(self.win_reward)+0.03)
            ax.set_xlim(0,1500)
            ax.set_xticks(np.arange(0,2000,300))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Expected Reward')
            ax.legend(fontsize = 8,loc = (0.01,0.1))
            ax.set_title(f'Gain Function for Decision Time\nAgent Mean,SD = {self.agent_means[i]},{self.agent_stds[i]}')#\n B = {B}')
            plt.show()