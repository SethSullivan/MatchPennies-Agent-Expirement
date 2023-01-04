import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os

class Subject():
    def __init__(self, **kwargs):
        self.num_blocks = kwargs.get('num_blocks',1)
        self.num_trials = kwargs.get('num_trials',1)
        self.num_control_trials = kwargs.get('num_control_trials',1)
        self.num_washout_trials = kwargs.get('num_washout_trials',1)                   
        # Control data
        self.reaction_time =           kwargs.get('reaction_time', np.zeros((self.num_control_trials)))
        self.reaction_movement_time =  kwargs.get('reaction_movement_time', np.zeros(self.num_control_trials))
        self.interval_trial_start =    kwargs.get('interval_trial_start', np.zeros((self.num_control_trials)))
        self.interval_reach_time =     kwargs.get('interval_reach_time', np.zeros((self.num_control_trials)))
        self.coincidence_trial_start = kwargs.get('coincidence_trial_start', np.zeros((self.num_control_trials)))
        self.coincidence_reach_time =  kwargs.get('coincidence_reach_time', np.zeros((self.num_control_trials)))
        # Washout data
        self.player_washout_decision_time =  kwargs.get('player_washout_decision_time',np.zeros((self.num_blocks,self.num_washout_trials)))
        self.player_washout_decision_array = kwargs.get('player_washout_decision_array',np.zeros((self.num_blocks,self.num_washout_trials)))
        self.player_washout_movement_time =  kwargs.get('player_washout_movement_time',np.zeros((self.num_blocks,self.num_washout_trials)))
        self.player_washout_reach_time =     kwargs.get('player_washout_reach_time',np.zeros((self.num_blocks,self.num_washout_trials)))
        self.agent_washout_decision_time =   kwargs.get('agent_washout_decision_time',np.zeros((self.num_blocks,self.num_washout_trials)))
        self.agent_washout_decision_array =  kwargs.get('agent_washout_decision_array',np.zeros((self.num_blocks,self.num_washout_trials)))
        self.agent_washout_movement_time =   kwargs.get('agent_washout_movement_time',np.zeros((self.num_blocks,self.num_washout_trials)))
        self.agent_washout_reach_time =      kwargs.get('agent_washout_reach_time',np.zeros((self.num_blocks,self.num_washout_trials)))       
        # Task data
        self.player_task_decision_time =  kwargs.get('player_task_decision_time',np.zeros((self.num_blocks,self.num_trials)))
        self.player_task_decision_array = kwargs.get('player_task_decision_array',np.zeros((self.num_blocks,self.num_trials)))
        self.player_task_movement_time =  kwargs.get('player_task_movement_time',np.zeros((self.num_blocks,self.num_trials)))
        self.player_task_reach_time =     kwargs.get('player_task_reach_time',np.zeros((self.num_blocks,self.num_trials)))
        self.agent_task_decision_time =   kwargs.get('agent_task_decision_time',np.zeros((self.num_blocks,self.num_trials)))
        self.agent_task_decision_array =  kwargs.get('agent_task_decision_array',np.zeros((self.num_blocks,self.num_trials)))
        self.agent_task_movement_time =   kwargs.get('agent_task_movement_time',np.zeros((self.num_blocks,self.num_trials)))
        self.agent_task_reach_time =      kwargs.get('agent_task_reach_time',np.zeros((self.num_blocks,self.num_trials))) 
        self.player_minus_agent_task_decision_time = self.player_task_decision_time - self.agent_task_decision_time
        
        self.num_stds_for_reaction_time = kwargs.get('num_stds_for_reaction_time',2)
        self.n = kwargs.get('cutoff_for_controls_calc',10)
        
        #------------------------Calculate Means and Stds----------------------------------------------------------------------------------------------------------
        # Control means
        self.reaction_time_mean = np.nanmean(self.reaction_time[self.n:])
        self.reaction_movement_time_mean = np.nanmean(self.reaction_movement_time[self.n:])
        self.reaction_plus_movement_time_mean = np.nanmean(self.reaction_time[self.n:] + self.reaction_movement_time[self.n:])
        self.interval_reach_time_mean = np.nanmean(self.interval_reach_time[self.n:])
        self.coincidence_reach_time_mean = np.nanmean(self.coincidence_reach_time[self.n:])
        # Control stds
        self.reaction_time_sd = np.nanstd(self.reaction_time[self.n:])
        self.reaction_movement_time_sd = np.nanstd(self.reaction_movement_time[self.n:])
        self.reaction_plus_movement_time_sd = np.nanstd(self.reaction_time[self.n:] + self.reaction_movement_time[self.n:])
        self.interval_reach_time_sd = np.nanstd(self.interval_reach_time[self.n:])
        self.coincidence_reach_time_sd = np.nanstd(self.coincidence_reach_time[self.n:])
        
        # CONSERVATIVE REACTION TIME
        self.reaction_time_minus_sd = self.reaction_time_mean - self.reaction_time_sd
        self.reaction_time_minus_2sd = self.reaction_time_mean - 2*self.reaction_time_sd
        
        # Task means and stds
        self.agent_task_reach_time_mean = np.nanmean(self.agent_task_reach_time,axis = 1)
        self.agent_task_reach_time_median = np.nanmedian(self.agent_task_reach_time,axis = 1)
        self.agent_task_reach_time_sd = np.nanstd(self.agent_task_reach_time,axis = 1)
        self.agent_task_decision_time_mean = np.nanmean(self.agent_task_decision_time, axis = 1)
        self.agent_task_decision_time_median = np.nanmedian(self.agent_task_decision_time, axis = 1)
        self.agent_task_decision_time_sd = np.nanstd(self.agent_task_decision_time, axis = 1)
        self.player_task_reach_time_mean = np.nanmean(self.player_task_reach_time,axis = 1)
        self.player_task_reach_time_median = np.nanmedian(self.player_task_reach_time,axis = 1)
        self.player_task_reach_time_sd = np.nanstd(self.player_task_reach_time,axis = 1)
        self.player_task_decision_time_mean = np.nanmean(self.player_task_decision_time,axis = 1)
        self.player_task_decision_time_median = np.nanmedian(self.player_task_decision_time,axis = 1)
        self.player_task_decision_time_sd = np.nanstd(self.agent_task_reach_time,axis = 1)
        self.player_task_movement_time_mean = np.nanmean(self.player_task_reach_time_mean - self.player_task_decision_time_mean)
        self.player_task_movement_time_median = np.nanmedian(self.player_task_reach_time_mean - self.player_task_decision_time_mean)
        self.player_task_movement_time_sd = np.nanstd(self.player_task_reach_time_mean - self.player_task_decision_time_mean)
        
        self.player_minus_agent_task_decision_time_mean = np.nanmean(self.player_minus_agent_task_decision_time, axis = 1)
        
        #------------------------------------------------------------------------------------------------------------------
        # Task Indecisions, Wins, Incorrects
        self.player_wins,self.player_indecisions,self.player_incorrects = self.calc_wins_indecisions_incorrects()
        self.perc_wins = (self.player_wins/self.num_trials)*100
        self.perc_indecisions = (self.player_indecisions/self.num_trials)*100
        self.perc_incorrects = (self.player_incorrects/self.num_trials)*100
        
        # Reach Times on Indecisions
        self.agent_task_decision_time_on_indecisions,self.player_task_reach_time_on_indecisions,\
            self.player_task_decision_time_on_indecisions,self.agent_mean_task_reach_time_on_indecisions,\
            self.player_mean_task_reach_time_on_indecisions,self.player_mean_task_decision_time_on_indecisions = self.decision_and_reach_times_on_indecisions()
                
        # Gamble and Reaction Calculations
        self.perc_reactions,self.perc_reaction_wins,self.perc_reaction_incorrects,self.perc_reaction_indecisions,\
            self.perc_wins_that_were_reactions,self.perc_indecisions_that_were_reactions,self.perc_incorrects_that_were_reactions,\
            self.reaction_decision_time_means, self.perc_gambles,self.perc_gamble_wins,self.perc_gamble_incorrects,self.perc_gamble_indecisions,\
            self.perc_wins_that_were_gambles,self.perc_indecisions_that_were_gambles,self.perc_incorrects_that_were_gambles, self.gamble_decision_time_means = self.reaction_gamble_calculations(self.num_stds_for_reaction_time)
        
        self.player_perc_wins_when_both_decide,self.player_wins_when_both_decide = self.wins_when_both_decide()
    def calc_wins_indecisions_incorrects(self):
        player_indecisions = np.zeros((self.num_blocks))
        player_wins = np.zeros((self.num_blocks))
        player_incorrects = np.zeros((self.num_blocks))
        for i in range(self.num_blocks):
            player_indecisions[i] = np.count_nonzero(self.player_task_decision_array[i,:] == 0)
            player_wins[i] = np.count_nonzero(np.logical_and(self.player_task_decision_array[i,:] == 1 , self.agent_task_decision_array[i,:] == 1))
            player_wins[i]+= np.count_nonzero(np.logical_and(self.player_task_decision_array[i,:] == -1 , self.agent_task_decision_array[i,:] == -1))
            player_wins[i]+= np.count_nonzero(np.logical_and(self.player_task_decision_array[i,:] == -1 , self.agent_task_decision_array[i,:] == 0))
            player_wins[i]+= np.count_nonzero(np.logical_and(self.player_task_decision_array[i,:] == 1 , self.agent_task_decision_array[i,:] == 0))
            player_incorrects[i] = np.count_nonzero(np.logical_and(self.player_task_decision_array[i,:] == 1 , self.agent_task_decision_array[i,:] == -1))
            player_incorrects[i] += np.count_nonzero(np.logical_and(self.player_task_decision_array[i,:] == -1 , self.agent_task_decision_array[i,:] == 1))
        return player_wins,player_indecisions,player_incorrects
    
    def decision_and_reach_times_on_indecisions(self):
        agent_task_decision_time_on_indecisions = np.zeros((self.num_blocks,self.num_trials))*np.nan
        player_task_reach_time_on_indecisions = np.zeros((self.num_blocks,self.num_trials))*np.nan
        player_left_time_on_indecisions = np.zeros((self.num_blocks,self.num_trials))*np.nan
        indecision_index = np.argwhere(self.player_task_reach_time>1500)

        c=0
        for j,k in indecision_index:
            agent_task_decision_time_on_indecisions[j,k] = self.agent_task_decision_time[j,k]
            player_task_reach_time_on_indecisions[j,k] = self.player_task_reach_time[j,k]
            player_left_time_on_indecisions[j,k] = self.player_task_decision_time[j,k]
            c+=1

        agent_mean_task_reach_time_on_indecisions = np.nanmean(agent_task_decision_time_on_indecisions,axis=1)
        player_mean_task_reach_time_on_indecisions = np.nanmean(player_task_reach_time_on_indecisions,axis=1)
        player_mean_left_time_on_indecisions = np.nanmean(player_left_time_on_indecisions,axis=1)
        return agent_task_decision_time_on_indecisions,player_task_reach_time_on_indecisions,player_left_time_on_indecisions,\
            agent_mean_task_reach_time_on_indecisions,player_mean_task_reach_time_on_indecisions,player_mean_left_time_on_indecisions
            
    def reaction_gamble_calculations(self,num_stds):
        # Gamble arrays
        gamble_decision_time = np.zeros((self.num_blocks,self.num_trials))*np.nan
        gamble_reach_target_time = np.zeros((self.num_blocks,self.num_trials))*np.nan
        agent_task_reach_time_gambles = np.zeros((self.num_blocks,self.num_trials))*np.nan
        agent_task_decision_time_gambles = np.zeros((self.num_blocks,self.num_trials))*np.nan
        # Reaction arrays
        reaction_decision_time = np.zeros((self.num_blocks,self.num_trials))*np.nan
        reaction_reach_target_time = np.zeros((self.num_blocks,self.num_trials))*np.nan
        agent_task_reach_time_reactions = np.zeros((self.num_blocks,self.num_trials))*np.nan
        agent_task_decision_time_reactions = np.zeros((self.num_blocks,self.num_trials))*np.nan

        # Wins, indecisiosn, incorrects arrays
        gamble_wins = np.zeros((self.num_blocks))
        perc_gamble_wins = np.zeros((self.num_blocks))
        gamble_indecisions = np.zeros((self.num_blocks))
        perc_gamble_indecisions = np.zeros((self.num_blocks))
        gamble_incorrects = np.zeros((self.num_blocks))
        perc_gamble_incorrects = np.zeros((self.num_blocks))
        reaction_wins = np.zeros((self.num_blocks))
        perc_reaction_wins = np.zeros((self.num_blocks))
        reaction_indecisions = np.zeros((self.num_blocks))
        perc_reaction_indecisions = np.zeros((self.num_blocks))
        reaction_incorrects = np.zeros((self.num_blocks))
        perc_reaction_incorrects = np.zeros((self.num_blocks))
        total_gambles = np.zeros((self.num_blocks))
        total_reactions = np.zeros((self.num_blocks))
        total_did_not_leave = np.zeros((self.num_blocks))

        temp_player_reaction_time =  self.reaction_time_mean - num_stds*self.reaction_time_sd
        gamble_index = np.argwhere((self.player_task_decision_time-self.agent_task_decision_time)<=temp_player_reaction_time)
        reaction_index = np.argwhere((self.player_task_decision_time-self.agent_task_decision_time)>temp_player_reaction_time)
        did_not_leave_start_index = np.argwhere(np.isnan(self.player_task_decision_time))
        for i,j in gamble_index:
            gamble_decision_time[i,j] = self.player_task_decision_time[i,j]
            gamble_reach_target_time[i,j] = self.player_task_reach_time[i,j]
            agent_task_reach_time_gambles[i,j] = self.agent_task_reach_time[i,j]
            agent_task_decision_time_gambles[i,j] = self.agent_task_decision_time[i,j]
            # Calculate gamble wins
            if self.player_task_decision_array[i,j] == 1 and (self.agent_task_decision_array[i,j] == 1 or self.agent_task_decision_array[i,j] == 0):
                gamble_wins[i] += 1
            elif self.player_task_decision_array[i,j] == -1 and (self.agent_task_decision_array[i,j] == -1 or self.agent_task_decision_array[i,j] == 0):
                gamble_wins[i] += 1
            elif self.player_task_decision_array[i,j] == 0:
                gamble_indecisions[i] += 1
            elif self.player_task_decision_array[i,j]*self.agent_task_decision_array[i,j] == -1:
                gamble_incorrects[i] += 1
            else:
                print('none')
            total_gambles[i]+=1
        for i,j in reaction_index:
            reaction_decision_time[i,j] = self.player_task_decision_time[i,j]
            reaction_reach_target_time[i,j] = self.player_task_reach_time[i,j]
            agent_task_reach_time_reactions[i,j] = self.agent_task_reach_time[i,j]
            agent_task_decision_time_reactions[i,j] = self.agent_task_decision_time[i,j]
            # Calculate reaction wins
            if self.player_task_decision_array[i,j] == 1 and (self.agent_task_decision_array[i,j] == 1 or self.agent_task_decision_array[i,j] == 0):
                reaction_wins[i] += 1
            elif self.player_task_decision_array[i,j] == -1 and (self.agent_task_decision_array[i,j] == -1 or self.agent_task_decision_array[i,j] == 0):
                reaction_wins[i] += 1
            elif self.player_task_decision_array[i,j] == 0:
                reaction_indecisions[i] += 1
            elif self.player_task_decision_array[i,j]*self.agent_task_decision_array[i,j] == -1:
                reaction_incorrects[i] += 1
            else:
                print('none')
            total_reactions[i]+=1
                
        for i,j in did_not_leave_start_index:
            reaction_indecisions[i]+=1
            total_did_not_leave[i]+=1
        perc_reactions = total_reactions/self.num_trials*100
        perc_reaction_wins = reaction_wins/total_reactions*100 # Array division
        perc_reaction_incorrects = reaction_incorrects/total_reactions*100
        perc_reaction_indecisions = reaction_indecisions/total_reactions*100

        perc_gambles = total_gambles/self.num_trials*100
        perc_gamble_wins = gamble_wins/total_gambles*100
        perc_gamble_incorrects = gamble_incorrects/total_gambles*100
        perc_gamble_indecisions = gamble_indecisions/total_gambles*100

        perc_wins_that_were_gambles = gamble_wins/self.player_wins *100
        perc_indecisions_that_were_gambles = gamble_indecisions/self.player_indecisions*100
        perc_incorrects_that_were_gambles = gamble_indecisions/self.player_incorrects*100

        perc_wins_that_were_reactions = reaction_wins/self.player_wins*100
        perc_indecisions_that_were_reactions = reaction_indecisions/self.player_indecisions*100
        perc_incorrects_that_were_reactions = reaction_indecisions/self.player_incorrects*100


        # get means
        gamble_decision_time_means = np.nanmean(gamble_decision_time, axis = 1 )
        reaction_decision_time_means = np.nanmean(reaction_decision_time, axis = 1 )
        agent_task_decision_time_gamble_means = np.nanmean(agent_task_decision_time_gambles, axis = 1)
        agent_task_decision_time_reaction_means = np.nanmean(agent_task_decision_time_reactions, axis = 1)

        # Not sure why I had this threshold in??
        # for i in range(self.num_blocks):
        #     if total_reactions[i]<10:
        #         perc_reaction_wins[i] = np.nan
        #         perc_reaction_incorrects[i] = np.nan
        #         perc_reaction_indecisions[i] = np.nan
        #         reaction_decision_time_means[i] = np.nan

        #     if total_gambles[i]<10:
        #         perc_gamble_wins[i] = np.nan
        #         perc_gamble_incorrects[i] = np.nan
        #         perc_gamble_indecisions[i] = np.nan
        #         gamble_decision_time_means[i] = np.nan
        
        return perc_reactions,perc_reaction_wins,perc_reaction_incorrects,perc_reaction_indecisions,\
            perc_wins_that_were_reactions,perc_indecisions_that_were_reactions,perc_incorrects_that_were_reactions, reaction_decision_time_means,\
            perc_gambles,perc_gamble_wins,perc_gamble_incorrects,perc_gamble_indecisions,\
            perc_wins_that_were_gambles,perc_indecisions_that_were_gambles,perc_incorrects_that_were_gambles, gamble_decision_time_means
   
    def wins_when_both_decide(self):
        # Get agent decision array
        player_both_reached_wins = np.zeros((self.num_blocks))
        perc_player_both_reached_wins = np.zeros((self.num_blocks))
        agent_both_reached_wins = np.zeros((self.num_blocks))
        for i in range(self.num_blocks):
            for j in range(self.num_trials):
                if self.agent_task_reach_time[i,j]>1500:
                    self.agent_task_decision_array[i,j] = 0
        # Get wins when both decide
        for i in range(self.num_blocks):
            for j in range(self.num_trials):
                if self.agent_task_decision_array[i,j]*self.player_task_decision_array[i,j] == 1:
                    player_both_reached_wins[i]+=1
                if self.agent_task_decision_array[i,j]*self.player_task_decision_array[i,j] == -1:
                    agent_both_reached_wins[i]+=1
            x = np.count_nonzero(self.player_task_decision_array[i,:]!=0)
            y = np.count_nonzero(self.agent_task_decision_array[i,:]!=0)
            if x!= 0 and y!= 0:
                total = np.count_nonzero(np.logical_and(self.player_task_decision_array[i,:]!=0,self.agent_task_decision_array[i,:]!=0))
                perc_player_both_reached_wins[i] = (player_both_reached_wins[i]/total)*100
        return perc_player_both_reached_wins,player_both_reached_wins
        
    
            
    
            
        
   







class Group():
    def __init__(self,objects):
        self.objects = objects
        
        # Control tasks, group means and uncertainties
        self.reaction_time_mean = np.nanmean(self.list_of_metrics('reaction_time_mean'))
        self.reaction_time_sd = np.nanmean(self.list_of_metrics('reaction_time_sd'))
        self.reaction_movement_time_mean = np.nanmean(self.list_of_metrics('reaction_movement_time_mean'))
        self.reaction_movement_time_sd = np.nanmean(self.list_of_metrics('reaction_movement_time_sd'))
        self.reaction_plus_movement_time_mean = np.nanmean(self.list_of_metrics('reaction_plus_movement_time_mean'))
        self.reaction_plus_movement_time_sd = np.nanmean(self.list_of_metrics('reaction_plus_movement_time_sd'))
        self.interval_reach_time_mean = np.nanmean(self.list_of_metrics('interval_reach_time_mean'))
        self.interval_reach_time_sd = np.nanmean(self.list_of_metrics('interval_reach_time_sd'))
        self.coincidence_reach_time_mean = np.nanmean(self.list_of_metrics('coincidence_reach_time_mean'))
        self.coincidence_reach_time_sd = np.nanmean(self.list_of_metrics('coincidence_reach_time_sd'))
        
        # Task mean, median, and sd, averaged across all participants
        self.agent_task_reach_time_mean = np.nanmean(self.list_of_metrics('agent_task_reach_time_mean'),axis = 0)
        self.agent_task_reach_time_median = np.nanmean(self.list_of_metrics('agent_task_reach_time_median'),axis = 0)
        self.agent_task_reach_time_sd = np.nanmean(self.list_of_metrics('agent_task_reach_time_sd'),axis = 0)
        self.agent_task_decision_time_mean = np.nanmean(self.list_of_metrics('agent_task_decision_time_mean'),axis =0)
        self.agent_task_decision_time_median = np.nanmean(self.list_of_metrics('agent_task_decision_time_median'),axis =0)
        self.agent_task_decision_time_sd = np.nanmean(self.list_of_metrics('agent_task_decision_time_sd'),axis =0)
        self.player_task_reach_time_mean = np.nanmean(self.list_of_metrics('player_task_reach_time_mean'),axis=0)
        self.player_task_reach_time_median = np.nanmean(self.list_of_metrics('player_task_reach_time_median'),axis=0)
        self.player_task_reach_time_sd = np.nanmean(self.list_of_metrics('player_task_reach_time_sd'),axis=0)
        self.player_task_decision_time_mean = np.nanmean(self.list_of_metrics('player_task_decision_time_mean'),axis=0)
        self.player_task_decision_time_median = np.nanmean(self.list_of_metrics('player_task_decision_time_median'),axis=0)
        self.player_task_decision_time_sd = np.nanmean(self.list_of_metrics('player_task_decision_time_sd'),axis = 0)
        
    def list_of_metrics(self,metric):
        return [getattr(o,metric) for o in self.objects]

