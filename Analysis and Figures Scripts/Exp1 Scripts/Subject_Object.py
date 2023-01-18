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
        self.reaction_time =           kwargs.get('reaction_time')
        self.reaction_movement_time =  kwargs.get('reaction_movement_time')
        self.interval_trial_start =    kwargs.get('interval_trial_start')
        self.interval_reach_time =     kwargs.get('interval_reach_time')
        self.coincidence_trial_start = kwargs.get('coincidence_trial_start')
        self.coincidence_reach_time =  kwargs.get('coincidence_reach_time')
        # Washout data
        self.player_washout_decision_time =  kwargs.get('player_washout_decision_time')
        self.player_washout_decision_array = kwargs.get('player_washout_decision_array')
        self.player_washout_movement_time =  kwargs.get('player_washout_movement_time')
        self.player_washout_reach_time =     kwargs.get('player_washout_reach_time')
        self.agent_washout_decision_time =   kwargs.get('agent_washout_decision_time')
        self.agent_washout_decision_array =  kwargs.get('agent_washout_decision_array')
        self.agent_washout_movement_time =   kwargs.get('agent_washout_movement_time')
        self.agent_washout_reach_time =      kwargs.get('agent_washout_reach_time')       
        # Task data
        self.player_task_decision_time =  kwargs.get('player_task_decision_time')
        self.player_task_decision_array = kwargs.get('player_task_decision_array')
        self.player_task_movement_time =  kwargs.get('player_task_movement_time')
        self.player_task_reach_time =     kwargs.get('player_task_reach_time')
        self.agent_task_decision_time =   kwargs.get('agent_task_decision_time')
        self.agent_task_decision_array =  kwargs.get('agent_task_decision_array')
        self.agent_task_movement_time =   kwargs.get('agent_task_movement_time')
        self.agent_task_reach_time =      kwargs.get('agent_task_reach_time')
        self.player_minus_agent_task_decision_time = self.player_task_decision_time - self.agent_task_decision_time
        
        self.num_stds_for_reaction_time = kwargs.get('num_stds_for_reaction_time',2)
        self.n = kwargs.get('cutoff_for_controls_calc',10)
        
        #------------------------Calculate Mean and Stds----------------------------------------------------------------------------------------------------------
        # Control mean
        self.reaction_time_mean = np.nanmean(self.reaction_time[self.n:])
        self.reaction_movement_time_mean = np.nanmean(self.reaction_movement_time[self.n:])
        self.reaction_plus_movement_time_mean = np.nanmean(self.reaction_time[self.n:] + self.reaction_movement_time[self.n:])
        self.interval_reach_time_mean = np.nanmean(self.interval_reach_time[self.n:])
        self.coincidence_reach_time_mean = np.nanmean(self.coincidence_reach_time[self.n:])
        # Control Medians
        self.reaction_time_median = np.nanmedian(self.reaction_time[self.n:])
        self.reaction_movement_time_median = np.nanmedian(self.reaction_movement_time[self.n:])
        self.reaction_plus_movement_time_median = np.nanmedian(self.reaction_time[self.n:] + self.reaction_movement_time[self.n:])
        self.interval_reach_time_median = np.nanmedian(self.interval_reach_time[self.n:])
        self.coincidence_reach_time_median = np.nanmedian(self.coincidence_reach_time[self.n:])
        # Control stds
        self.reaction_time_sd = np.nanstd(self.reaction_time[self.n:])
        self.reaction_movement_time_sd = np.nanstd(self.reaction_movement_time[self.n:])
        self.reaction_plus_movement_time_sd = np.nanstd(self.reaction_time[self.n:] + self.reaction_movement_time[self.n:])
        self.interval_reach_time_sd = np.nanstd(self.interval_reach_time[self.n:])
        self.coincidence_reach_time_sd = np.nanstd(self.coincidence_reach_time[self.n:])
        
        # CONSERVATIVE REACTION TIME
        self.reaction_time_minus_sd = self.reaction_time_mean - self.reaction_time_sd
        self.reaction_time_minus_2sd = self.reaction_time_mean - 2*self.reaction_time_sd
        
        # Task mean and stds
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
        self.player_task_decision_time_sd = np.nanstd(self.player_task_reach_time,axis = 1)
        self.player_task_movement_time_mean = np.nanmean(self.player_task_reach_time_mean - self.player_task_decision_time_mean)
        self.player_task_movement_time_median = np.nanmedian(self.player_task_reach_time_mean - self.player_task_decision_time_mean)
        self.player_task_movement_time_sd = np.nanstd(self.player_task_reach_time_mean - self.player_task_decision_time_mean)
        
        self.player_minus_agent_task_decision_time_mean = np.nanmean(self.player_minus_agent_task_decision_time, axis = 1)
        
        #------------------------------------------------------------------------------------------------------------------
        # Task Indecisions, Wins, Incorrects
        self.player_wins,self.player_indecisions,self.player_incorrects = self.calc_wins_indecisions_incorrects()
        self.player_perc_wins = (self.player_wins/self.num_trials)*100
        self.player_perc_indecisions = (self.player_indecisions/self.num_trials)*100
        self.player_perc_incorrects = (self.player_incorrects/self.num_trials)*100
        
        # Decision and Reach  Times on Indecisions
        self.decision_and_reach_times_on_indecisions()
                
        # Gamble and Reaction Calculations
        self.reaction_gamble_calculations(self.num_stds_for_reaction_time)
        
        # Wins when both decide
        self.wins_when_both_decide()
        
        # Binned metrics
        self.binned_metrics()
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
        self.agent_task_decision_time_on_indecisions = np.zeros((self.num_blocks,self.num_trials))*np.nan
        self.player_task_reach_time_on_indecisions = np.zeros((self.num_blocks,self.num_trials))*np.nan
        self.player_left_time_on_indecisions = np.zeros((self.num_blocks,self.num_trials))*np.nan
        indecision_index = np.argwhere(self.player_task_reach_time>1500)

        c=0
        for j,k in indecision_index:
            self.agent_task_decision_time_on_indecisions[j,k] = self.agent_task_decision_time[j,k]
            self.player_task_reach_time_on_indecisions[j,k] = self.player_task_reach_time[j,k]
            self.player_left_time_on_indecisions[j,k] = self.player_task_decision_time[j,k]
            c+=1

        self.agent_mean_task_reach_time_on_indecisions = np.nanmean(self.agent_task_decision_time_on_indecisions,axis=1)
        self.player_mean_task_reach_time_on_indecisions = np.nanmean(self.player_task_reach_time_on_indecisions,axis=1)
        self.player_mean_left_time_on_indecisions = np.nanmean(self.player_left_time_on_indecisions,axis=1)
            
    def reaction_gamble_calculations(self,num_stds):
        # Gamble arrays
        self.gamble_decision_time = np.zeros((self.num_blocks,self.num_trials))*np.nan
        self.gamble_reach_target_time = np.zeros((self.num_blocks,self.num_trials))*np.nan
        self.agent_task_reach_time_gambles = np.zeros((self.num_blocks,self.num_trials))*np.nan
        self.agent_task_decision_time_gambles = np.zeros((self.num_blocks,self.num_trials))*np.nan
        # Reaction arrays
        self.reaction_decision_time = np.zeros((self.num_blocks,self.num_trials))*np.nan
        self.reaction_reach_target_time = np.zeros((self.num_blocks,self.num_trials))*np.nan
        self.agent_task_reach_time_reactions = np.zeros((self.num_blocks,self.num_trials))*np.nan
        self.agent_task_decision_time_reactions = np.zeros((self.num_blocks,self.num_trials))*np.nan

        # Wins, indecisiosn, incorrects arrays
        self.gamble_wins = np.zeros((self.num_blocks))
        self.perc_gamble_wins = np.zeros((self.num_blocks))
        self.gamble_indecisions = np.zeros((self.num_blocks))
        self.perc_gamble_indecisions = np.zeros((self.num_blocks))
        self.gamble_incorrects = np.zeros((self.num_blocks))
        self.perc_gamble_incorrects = np.zeros((self.num_blocks))
        self.reaction_wins = np.zeros((self.num_blocks))
        self.perc_reaction_wins = np.zeros((self.num_blocks))
        self.reaction_indecisions = np.zeros((self.num_blocks))
        self.perc_reaction_indecisions = np.zeros((self.num_blocks))
        self.reaction_incorrects = np.zeros((self.num_blocks))
        self.perc_reaction_incorrects = np.zeros((self.num_blocks))
        self.total_gambles = np.zeros((self.num_blocks))
        self.total_reactions = np.zeros((self.num_blocks))
        self.total_did_not_leave = np.zeros((self.num_blocks))

        temp_player_reaction_time =  self.reaction_time_mean - num_stds*self.reaction_time_sd
        gamble_index = np.argwhere((self.player_task_decision_time-self.agent_task_decision_time)<=temp_player_reaction_time)
        reaction_index = np.argwhere((self.player_task_decision_time-self.agent_task_decision_time)>temp_player_reaction_time)
        did_not_leave_start_index = np.argwhere(np.isnan(self.player_task_decision_time))
        for i,j in gamble_index:
            self.gamble_decision_time[i,j] = self.player_task_decision_time[i,j]
            self.gamble_reach_target_time[i,j] = self.player_task_reach_time[i,j]
            self.agent_task_reach_time_gambles[i,j] = self.agent_task_reach_time[i,j]
            self.agent_task_decision_time_gambles[i,j] = self.agent_task_decision_time[i,j]
            # Calculate gamble wins
            if self.player_task_decision_array[i,j] == 1 and (self.agent_task_decision_array[i,j] == 1 or self.agent_task_decision_array[i,j] == 0):
                self.gamble_wins[i] += 1
            elif self.player_task_decision_array[i,j] == -1 and (self.agent_task_decision_array[i,j] == -1 or self.agent_task_decision_array[i,j] == 0):
                self.gamble_wins[i] += 1
            elif self.player_task_decision_array[i,j] == 0:
                self.gamble_indecisions[i] += 1
            elif self.player_task_decision_array[i,j]*self.agent_task_decision_array[i,j] == -1:
                self.gamble_incorrects[i] += 1
            else:
                print('none')
            self.total_gambles[i]+=1
        for i,j in reaction_index:
            self.reaction_decision_time[i,j] = self.player_task_decision_time[i,j]
            self.reaction_reach_target_time[i,j] = self.player_task_reach_time[i,j]
            self.agent_task_reach_time_reactions[i,j] = self.agent_task_reach_time[i,j]
            self.agent_task_decision_time_reactions[i,j] = self.agent_task_decision_time[i,j]
            # Calculate reaction wins
            if self.player_task_decision_array[i,j] == 1 and (self.agent_task_decision_array[i,j] == 1 or self.agent_task_decision_array[i,j] == 0):
                self.reaction_wins[i] += 1
            elif self.player_task_decision_array[i,j] == -1 and (self.agent_task_decision_array[i,j] == -1 or self.agent_task_decision_array[i,j] == 0):
                self.reaction_wins[i] += 1
            elif self.player_task_decision_array[i,j] == 0:
                self.reaction_indecisions[i] += 1
            elif self.player_task_decision_array[i,j]*self.agent_task_decision_array[i,j] == -1:
                self.reaction_incorrects[i] += 1
            else:
                print('none')
            self.total_reactions[i]+=1
                
        for i,j in did_not_leave_start_index:
            self.reaction_indecisions[i]+=1
            self.total_did_not_leave[i]+=1
        self.perc_reactions = (self.total_reactions/self.num_trials)*100
        self.perc_reaction_wins = (self.reaction_wins/self.total_reactions)*100 # Array division
        self.perc_reaction_incorrects = (self.reaction_incorrects/self.total_reactions)*100
        self.perc_reaction_indecisions = (self.reaction_indecisions/self.total_reactions)*100

        self.perc_gambles = (self.total_gambles/self.num_trials)*100
        self.perc_gamble_wins = (self.gamble_wins/self.total_gambles)*100
        self.perc_gamble_incorrects = (self.gamble_incorrects/self.total_gambles)*100
        self.perc_gamble_indecisions = (self.gamble_indecisions/self.total_gambles)*100

        self.perc_wins_that_were_gambles = (self.gamble_wins/self.player_wins) *100
        self.perc_indecisions_that_were_gambles = (self.gamble_indecisions/self.player_indecisions)*100
        self.perc_incorrects_that_were_gambles = (self.gamble_incorrects/self.player_incorrects)*100

        self.perc_wins_that_were_reactions = (self.reaction_wins/self.player_wins)*100
        self.perc_indecisions_that_were_reactions = (self.reaction_indecisions/self.player_indecisions)*100
        self.perc_incorrects_that_were_reactions = (self.reaction_incorrects/self.player_incorrects)*100

        # get mean
        self.gamble_decision_time_mean = np.nanmean(self.gamble_decision_time, axis = 1 )
        self.gamble_decision_time_sd = np.nanstd(self.gamble_decision_time, axis = 1 )
        self.reaction_decision_time_mean = np.nanmean(self.reaction_decision_time, axis = 1 )
        self.reaction_decision_time_sd = np.nanstd(self.reaction_decision_time, axis = 1 )
        self.agent_task_decision_time_gamble_mean = np.nanmean(self.agent_task_decision_time_gambles, axis = 1)
        self.agent_task_decision_time_reaction_mean = np.nanmean(self.agent_task_decision_time_reactions, axis = 1)

        # Not sure why I had this threshold in??
        # for i in range(self.num_blocks):
        #     if total_reactions[i]<10:
        #         perc_reaction_wins[i] = np.nan
        #         perc_reaction_incorrects[i] = np.nan
        #         perc_reaction_indecisions[i] = np.nan
        #         reaction_decision_time_mean[i] = np.nan

        #     if total_gambles[i]<10:
        #         perc_gamble_wins[i] = np.nan
        #         perc_gamble_incorrects[i] = np.nan
        #         perc_gamble_indecisions[i] = np.nan
        #         self.gamble_decision_time_mean[i] = np.nan
        return
        
    def wins_when_both_decide(self):
        # Get agent decision array
        self.player_both_reached_wins = np.zeros((self.num_blocks))
        self.player_perc_both_reached_wins = np.zeros((self.num_blocks))
        self.agent_both_reached_wins = np.zeros((self.num_blocks))
        # Get wins when both decide
        for i in range(self.num_blocks):
            for j in range(self.num_trials):
                if self.agent_task_decision_array[i,j]*self.player_task_decision_array[i,j] == 1:
                    self.player_both_reached_wins[i]+=1
                if self.agent_task_decision_array[i,j]*self.player_task_decision_array[i,j] == -1:
                    self.agent_both_reached_wins[i]+=1
            x = np.count_nonzero(self.player_task_decision_array[i,:]!=0)
            y = np.count_nonzero(self.agent_task_decision_array[i,:]!=0)
            if x!= 0 and y!= 0:
                total = np.count_nonzero(np.logical_and(self.player_task_decision_array[i,:]!=0,self.agent_task_decision_array[i,:]!=0))
                self.player_perc_both_reached_wins[i] = (self.player_both_reached_wins[i]/total)*100
                
    def binned_metrics(self,bin_start = 800,bin_end = 1400, bin_size = 50,cut_off_threshold = 30):
        self.bins = np.arange(bin_start,bin_end,bin_size)
        self.bin_length_each_condition                    = np.zeros((len(self.bins)-1,self.num_blocks))
        self.binned_player_task_decision_times            = np.zeros((len(self.bins)-1,self.num_blocks,self.num_trials))*np.nan 
        self.binned_player_task_decision_array            = np.zeros((len(self.bins)-1,self.num_blocks,self.num_trials))*np.nan 
        self.binned_agent_task_decision_times             = np.zeros((len(self.bins)-1,self.num_blocks,self.num_trials))*np.nan 
        self.binned_player_minus_agent_task_decision_time = np.zeros((len(self.bins)-1,self.num_blocks,self.num_trials))*np.nan 
        self.binned_agent_task_decision_array             = np.zeros((len(self.bins)-1,self.num_blocks,self.num_trials))*np.nan 
        self.binned_player_wins                           = np.zeros((len(self.bins)-1,self.num_blocks))
        self.binned_player_indecisions                    = np.zeros((len(self.bins)-1,self.num_blocks))
        self.binned_player_incorrects                     = np.zeros((len(self.bins)-1,self.num_blocks))
        self.mean_binned_player_wins                      = np.zeros((len(self.bins)-1,self.num_blocks))
        self.mean_binned_player_indecisions               = np.zeros((len(self.bins)-1,self.num_blocks))
        self.mean_binned_player_incorrects                = np.zeros((len(self.bins)-1,self.num_blocks))
        for b in range(len(self.bins)-1):
            bin_index = np.argwhere((self.bins[b] < self.agent_task_decision_time) & (self.agent_task_decision_time < self.bins[b+1]))
            for i,j in bin_index:
                self.bin_length_each_condition[b,i]+=1
                self.binned_player_task_decision_times[b,i,j] = self.player_task_decision_time[i,j] 
                self.binned_player_task_decision_array[b,i,j] = self.player_task_decision_array[i,j]
                self.binned_agent_task_decision_times[b,i,j] = self.agent_task_decision_time[i,j]
                self.binned_agent_task_decision_array[b,i,j] = self.agent_task_decision_array[i,j] 
                if ((self.player_task_decision_array[i,j]*self.agent_task_decision_array[i,j] == 1) or (self.player_task_decision_array[i,j] != 0 and self.agent_task_decision_array[i,j] == 0)):
                    self.binned_player_wins[b,i] += 1
                elif self.player_task_decision_array[i,j] == 0:
                    self.binned_player_indecisions[b,i] += 1
                elif (self.player_task_decision_array[i,j]*self.agent_task_decision_array[i,j] == -1):
                    self.binned_player_incorrects[b,i] += 1        
                
        self.binned_player_minus_agent_task_decision_time = self.binned_player_task_decision_times - self.binned_agent_task_decision_times
        # Get percentages based on bin length     
        self.perc_binned_player_wins = (self.binned_player_wins/self.bin_length_each_condition)*100
        self.perc_binned_player_indecisions = (self.binned_player_indecisions/self.bin_length_each_condition)*100
        self.perc_binned_player_incorrects = (self.binned_player_incorrects/self.bin_length_each_condition)*100
        # Calculate mean across all trials
        self.binned_player_task_decision_times_mean = np.nanmean(self.binned_player_task_decision_times,axis=2) # Mean for each bin, each condition
        self.binned_player_minus_agent_task_decision_time_mean = np.nanmean(self.binned_player_minus_agent_task_decision_time,axis=2)
        
        # Cut off at threshold
        mask = self.bin_length_each_condition>cut_off_threshold
        self.perc_binned_player_wins_cutoff = self.perc_binned_player_wins*mask
        self.perc_binned_player_indecisions_cutoff = self.perc_binned_player_indecisions*mask
        self.perc_binned_player_incorrects_cutoff = self.perc_binned_player_incorrects*mask
        self.binned_player_minus_agent_task_decision_time_mean_cutoff = self.binned_player_minus_agent_task_decision_time_mean*mask
        self.binned_player_task_decision_times_mean_cutoff = self.binned_player_task_decision_times_mean*mask
        
class Group():
    def __init__(self, objects,**kwargs):
        self.objects = objects
        self.num_blocks = kwargs.get('num_blocks',6)
        self.num_trials = kwargs.get('num_trials',80)
        self.bin_cutoff_threshold = kwargs.get('bin_cutoff_threshold',30)
        # Control tasks, group mean and uncertainties
        self.reaction_time_mean = np.nanmean(self.combine_all_subjects('reaction_time_mean'))
        self.reaction_time_median = np.nanmedian(self.combine_all_subjects('reaction_time_mean'))
        self.reaction_time_sd = np.nanmean(self.combine_all_subjects('reaction_time_sd'))
        self.reaction_movement_time_mean = np.nanmean(self.combine_all_subjects('reaction_movement_time_mean'))
        self.reaction_movement_time_median = np.nanmedian(self.combine_all_subjects('reaction_movement_time_mean'))
        self.reaction_movement_time_sd = np.nanmean(self.combine_all_subjects('reaction_movement_time_sd'))
        self.reaction_plus_movement_time_mean = np.nanmean(self.combine_all_subjects('reaction_plus_movement_time_mean'))
        self.reaction_plus_movement_time_median = np.nanmedian(self.combine_all_subjects('reaction_plus_movement_time_mean'))
        self.reaction_plus_movement_time_sd = np.nanmean(self.combine_all_subjects('reaction_plus_movement_time_sd'))
        self.interval_reach_time_mean = np.nanmean(self.combine_all_subjects('interval_reach_time_mean'))
        self.interval_reach_time_median = np.nanmedian(self.combine_all_subjects('interval_reach_time_mean'))
        self.interval_reach_time_sd = np.nanmean(self.combine_all_subjects('interval_reach_time_sd'))
        self.coincidence_reach_time_mean = np.nanmean(self.combine_all_subjects('coincidence_reach_time_mean'))
        self.coincidence_reach_time_median = np.nanmedian(self.combine_all_subjects('coincidence_reach_time_mean'))
        self.coincidence_reach_time_sd = np.nanmean(self.combine_all_subjects('coincidence_reach_time_sd'))
        
        # Task mean, median, and sd, averaged across all participants (should be shape (6,) ) 
        self.agent_task_reach_time_mean = np.nanmean(self.combine_all_subjects('agent_task_reach_time_mean'),axis = 0)
        self.agent_task_reach_time_median = np.nanmean(self.combine_all_subjects('agent_task_reach_time_median'),axis = 0)
        self.agent_task_reach_time_sd = np.nanmean(self.combine_all_subjects('agent_task_reach_time_sd'),axis = 0)
        self.agent_task_decision_time_mean = np.nanmean(self.combine_all_subjects('agent_task_decision_time_mean'),axis = 0)
        self.agent_task_decision_time_median = np.nanmean(self.combine_all_subjects('agent_task_decision_time_median'),axis = 0)
        self.agent_task_decision_time_sd = np.nanmean(self.combine_all_subjects('agent_task_decision_time_sd'),axis = 0)
        self.player_task_reach_time_mean = np.nanmean(self.combine_all_subjects('player_task_reach_time_mean'),axis = 0)
        self.player_task_reach_time_median = np.nanmean(self.combine_all_subjects('player_task_reach_time_median'),axis = 0)
        self.player_task_reach_time_sd = np.nanmean(self.combine_all_subjects('player_task_reach_time_sd'),axis = 0)
        self.player_task_decision_time_mean = np.nanmean(self.combine_all_subjects('player_task_decision_time_mean'),axis = 0)
        self.player_task_decision_time_median = np.nanmean(self.combine_all_subjects('player_task_decision_time_median'),axis = 0)
        self.player_task_decision_time_sd = np.nanmean(self.combine_all_subjects('player_task_decision_time_sd'),axis = 0)
        
        self.all_player_task_decision_times_each_condition = self.concatenate_across_subjects('player_task_decision_time')
        self.all_agent_task_decision_times_each_condition = self.concatenate_across_subjects('agent_task_decision_time')
        self.all_player_mean_decision_time_each_condition = np.nanmean(self.all_player_task_decision_times_each_condition,axis=1)
        self.all_player_median_decision_time_each_condition = np.nanmedian(self.all_player_task_decision_times_each_condition,axis=1)
        self.all_agent_mean_decision_time_each_condition = np.nanmean(self.all_agent_task_decision_times_each_condition,axis=1)
        
        self.all_player_task_reach_times_each_condition = self.concatenate_across_subjects('player_task_reach_time')
        self.all_agent_task_reach_times_each_condition = self.concatenate_across_subjects('agent_task_reach_time')
        self.all_player_mean_reach_time_each_condition = np.nanmean(self.all_player_task_reach_times_each_condition,axis=1)
        self.all_agent_mean_reach_time_each_condition = np.nanmean(self.all_agent_task_reach_times_each_condition,axis=1)
        
        # BINNING ----------------------------------------------------------------------------------------
        # Binned mean across all participants
        self.perc_binned_player_wins_mean = np.nanmean(self.combine_all_subjects('perc_binned_player_wins'),axis = 0)
        self.perc_binned_player_indecisions_mean = np.nanmean(self.combine_all_subjects('perc_binned_player_indecisions'),axis = 0)
        self.perc_binned_player_incorrects_mean = np.nanmean(self.combine_all_subjects('perc_binned_player_incorrects'),axis = 0)
        self.binned_player_minus_agent_task_decision_time_mean = np.nanmean(self.combine_all_subjects('binned_player_minus_agent_task_decision_time_mean'),axis = 0)
        self.binned_player_task_decision_times_mean = np.nanmean(self.combine_all_subjects('binned_player_task_decision_times_mean'),axis = 0)
        
        # Combine subjects into array
        self.bin_length_each_subject_each_condition = self.combine_all_subjects('bin_length_each_condition')
        self.bin_length_each_condition = np.sum(self.bin_length_each_subject_each_condition,axis=0)
        self.binned_player_minus_agent_task_decision_time = self.combine_all_subjects('binned_player_minus_agent_task_decision_time')
        self.perc_binned_player_wins = self.combine_all_subjects('perc_binned_player_wins')
        self.perc_binned_player_indecisions = self.combine_all_subjects('perc_binned_player_indecisions')
        self.perc_binned_player_incorrects = self.combine_all_subjects('perc_binned_player_incorrects')
        
        self.bin_threshold()
        
    def combine_all_subjects(self,metric):
        '''
        List comprehension into np array to put the subjects at index 0
        '''
        return np.array([getattr(o,metric) for o in self.objects])
    
    def concatenate_across_subjects(self,metric):
        '''
        Flattens out the subject dimension to get array of all the subjects 
        
        Usually used for group distributions
        '''
        arr = self.combine_all_subjects(metric)
        temp = np.swapaxes(arr,0,1)
        ans = np.reshape(temp,(self.num_blocks,-1))
        return ans
    
    def find_subject(self,metric,comparison_num,comparison_direction):
        '''
        Used to find the subject who's specific value is greater or less than the inputted comparison metric
        '''
        metrics = self.combine_all_subjects(metric)
        for i,m in enumerate(metrics):
            if comparison_direction == 'greater than':
                if m > comparison_num:
                    print(f'Sub{i+1}')
            if comparison_direction == 'less than':
                if m < comparison_num:
                    print(f'Sub{i+1}')
    def bin_threshold(self):
        '''
        Cut off bins so they only show if the bin has over the amount of the certain threshold
        '''
        # Cut off at threshold
        mask = self.bin_length_each_condition>self.bin_cutoff_threshold
        print(mask.shape)
        self.perc_binned_player_wins_mean_cutoff = self.perc_binned_player_wins_mean*mask
        self.perc_binned_player_indecisions_mean_cutoff = self.perc_binned_player_indecisions_mean*mask
        self.perc_binned_player_incorrects_mean_cutoff = self.perc_binned_player_incorrects_mean*mask
        self.binned_player_minus_agent_task_decision_time_mean_cutoff = self.binned_player_minus_agent_task_decision_time_mean*mask
        self.binned_player_task_decision_times_mean_cutoff = self.binned_player_task_decision_times_mean*mask   