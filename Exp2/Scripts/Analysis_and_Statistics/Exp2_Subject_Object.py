import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
class Subject():
    def __init__(self,**kwargs):
        self.num_blocks                             = kwargs.get('num_blocks')
        self.num_trials                             = kwargs.get('num_trials')
        self.num_reaction_blocks                    = kwargs.get('num_reaction_blocks')
        self.num_reaction_trials                    = kwargs.get('num_reaction_trials')
        
        self.num_timing_trials                      = kwargs.get('num_timing_trials')
        self.select_trials                          = kwargs.get('select_trials') 
        
        if self.select_trials != 'All Trials':
            self.num_trials = self.num_trials//2
        # Reaction Data
        self.reaction_trial_type_array              = kwargs.get('reaction_trial_type_array')
        self.reaction_time                          = kwargs.get('reaction_time')
        self.reaction_movement_time                 = kwargs.get('reaction_movement_time')
        self.reaction_plus_movement_times           = kwargs.get('reaction_plus_movement_time')
        # Timing Data
        self.interval_trial_start                   = kwargs.get('interval_trial_start')
        self.interval_reach_time                    = kwargs.get('interval_reach_time')
        self.coincidence_trial_start                = kwargs.get('coincidence_trial_start')
        self.coincidence_reach_time                 = kwargs.get('coincidence_reach_time')
        
        # Task data
        self.player_task_leave_time                = self.slice_array(kwargs.get('player_task_leave_time'))
        self.player_task_decision_array            = self.slice_array(kwargs.get('player_task_decision_array'))
        self.player_task_movement_time             = self.slice_array(kwargs.get('player_task_movement_time'))
        self.player_task_reach_time                = self.slice_array(kwargs.get('player_task_reach_time'))
        self.agent_task_leave_time                 = self.slice_array(kwargs.get('agent_task_leave_time'))
        self.agent_task_decision_array             = self.slice_array(kwargs.get('agent_task_decision_array'))
        self.agent_task_movement_time              = self.slice_array(kwargs.get('agent_task_movement_time'))
        self.agent_task_reach_time                 = self.slice_array(kwargs.get('agent_task_reach_time'))
        self.player_minus_agent_task_leave_time = self.player_task_leave_time - self.agent_task_leave_time
        
        
        self.num_stds_for_reaction_time = kwargs.get('num_stds_for_reaction_time')
        self.n = kwargs.get('cutoff_for_controls_calc',10)
        # self.reaction_time_700_mask = self.reaction_time < 700
        # self.reaction_time = self.mask_array(self.reaction_time,self.reaction_time_700_mask)
        
        self.win_mask = np.logical_or(self.player_task_decision_array*self.agent_task_decision_array == 1, np.logical_and(self.player_task_decision_array!=0, self.agent_task_decision_array==0))
        self.indecision_mask = self.player_task_decision_array == 0 
        self.incorrect_mask = self.player_task_decision_array*self.agent_task_decision_array == -1
        self.both_decided_mask = abs(self.player_task_decision_array*self.agent_task_decision_array) == 1
        
        self.create_result_of_trial_array()
        
    def slice_array(self,arr):
        if isinstance(arr,np.ndarray):
            if self.select_trials == 'First Half':
                return arr[:,:self.num_trials]
            elif self.select_trials == 'Second Half':
                return arr[:,self.num_trials:]
            elif self.select_trials == 'All Trials':
                return arr
            else:
                raise Exception('select_trials should be First Half, Second Half, or All Trials')
        else:
            return arr
        
    def mask_array(self,arr,mask):
        '''
        Applies the mask to the array then replaces the 0s with nans
        '''
        new_arr = arr*mask # Apply mask
        new_arr[~mask] = np.nan # Replace the 0s from the mask with np nan
        return new_arr
    
    def create_result_of_trial_array(self):
        '''
        1: Win
        2: Indecision
        3: Incorrect
        '''
        self.trial_results = np.zeros((self.num_blocks,self.num_trials))*np.nan
        self.trial_results[self.win_mask] = 1
        self.trial_results[self.indecision_mask] = 2
        self.trial_results[self.incorrect_mask] = 3  
        
    # def remove_reaction_time_nans(self,arr=None,mask=None):
    #     '''
    #     This actually removes the values of the array, as opposed to replacing False values with 
    #     NanN.
        
    #     Need to do this to plot boxplots I think 
    #     '''
    #     gamble_nanmask = np.isnan(self.gamble_reaction_time_all)
    #     self.gamble_reaction_time_only_gamble        = self.gamble_reaction_time_all[2,:][~gamble_nanmask[2,:]]
    #     self.gamble_reaction_time_mixed              = self.gamble_reaction_time_all[0,:][~gamble_nanmask[0,:]]
        
    #     react_nanmask = np.isnan(self.react_reaction_time_all)
    #     self.react_reaction_time_only_react         = self.react_reaction_time_all[1,:][~react_nanmask[1,:]]
    #     self.react_reaction_time_mixed              = self.react_reaction_time_all[0,:][~react_nanmask[0,:]]
        
    def analyze_data(self):
        
        # Parse Reaction TIme Data
        self.parse_reaction_task_trials()
        self.calculate_reaction_repeats_alternates()
        # self.remove_reaction_time_nans()
        self.adjusted_player_reaction_time    = np.nanmean(self.react_reaction_time_only_react) - \
                                                self.num_stds_for_reaction_time*np.nanstd(self.react_reaction_time_only_react) # CONSERVATIVE REACTION TIME
        #------------------------------------------------------------------------------------------------------------------
        # Task Indecision, Wins, Incorrects
        self.calc_wins_indecisions_incorrects()
        
        # Decision and Reach  Times on Inleaves
        self.leave_and_reach_times_on_wins_incorrects_indecisions()
                
        # Gamble and Reaction Calculations
        # self.reaction_gamble_calculations()
        self.reaction_gamble_calculations()
        
        # Wins when both decide
        self.wins_when_both_decide()
        
        # Binned metrics
        self.binned_metrics()
        
    def parse_reaction_task_trials(self):
        self.reaction_gamble_mask                 = self.reaction_trial_type_array == 0
        self.reaction_react_mask                  = self.reaction_trial_type_array == 1

        # * Reaction Time        
        self.gamble_reaction_time_all    = self.mask_array(self.reaction_time,self.reaction_gamble_mask) 
        self.react_reaction_time_all     = self.mask_array(self.reaction_time,self.reaction_react_mask) 
        
        self.reaction_time_mixed = self.reaction_time[0,:]
        
        self.gamble_reaction_time_only_gamble        = self.gamble_reaction_time_all[2,:]
        self.gamble_reaction_time_mixed              = self.gamble_reaction_time_all[0,:]
        
        self.react_reaction_time_only_react         = self.react_reaction_time_all[1,:]
        self.react_reaction_time_mixed              = self.react_reaction_time_all[0,:]
        
        # * Movement time
        self.gamble_movement_time_all    = self.mask_array(self.reaction_movement_time,self.reaction_gamble_mask) 
        self.react_movement_time_all     = self.mask_array(self.reaction_movement_time,self.reaction_react_mask) 
        
        self.movement_times_mixed = self.reaction_movement_time[0,:]
        
        self.gamble_movement_time_only_gamble        = self.gamble_movement_time_all[2,:]
        self.gamble_movement_time_mixed              = self.gamble_movement_time_all[0,:]
        
        self.react_movement_time_only_react         = self.react_movement_time_all[1,:]
        self.react_movement_time_mixed              = self.react_movement_time_all[0,:]
        
    def calculate_reaction_repeats_alternates(self):
        # Get masks
        self.react_repeat_mask  = np.full(self.num_reaction_trials,False)
        self.react_switch_mask  = np.full(self.num_reaction_trials,False)
        self.gamble_repeat_mask = np.full(self.num_reaction_trials,False)
        self.gamble_switch_mask = np.full(self.num_reaction_trials,False)
        for i in range(self.num_reaction_trials-1):
            if (self.reaction_trial_type_array[0,i]==1 and self.reaction_trial_type_array[0,i+1] == 1):
                self.react_repeat_mask[i+1]  = True
                self.react_switch_mask[i+1]  = False
                self.gamble_repeat_mask[i+1] = False
                self.gamble_switch_mask[i+1] = False
            elif (self.reaction_trial_type_array[0,i]==1 and self.reaction_trial_type_array[0,i+1]==0):
                self.react_repeat_mask[i+1]  = False
                self.react_switch_mask[i+1]  = False
                self.gamble_repeat_mask[i+1] = False
                self.gamble_switch_mask[i+1] = True   
            elif (self.reaction_trial_type_array[0,i]==0 and self.reaction_trial_type_array[0,i+1]==1):
                self.react_repeat_mask[i+1]  = False
                self.react_switch_mask[i+1]  = True
                self.gamble_repeat_mask[i+1] = False
                self.gamble_switch_mask[i+1] = False
            elif (self.reaction_trial_type_array[0,i]==0 and self.reaction_trial_type_array[0,i+1]==0):
                self.react_repeat_mask[i+1]  = False
                self.react_switch_mask[i+1]  = False
                self.gamble_repeat_mask[i+1] = True
                self.gamble_switch_mask[i+1] = False
        
        # Get the reaction times of gambles on repeats and switchs
        self.gamble_reaction_time_repeat = self.mask_array(self.reaction_time_mixed,self.gamble_repeat_mask)
        self.gamble_reaction_time_switch = self.mask_array(self.reaction_time_mixed,self.gamble_switch_mask)
        # Get the reaction times of reactions on repeats and mixeds
        self.react_reaction_time_repeat  = self.mask_array(self.reaction_time_mixed,self.react_repeat_mask)
        self.react_reaction_time_switch  = self.mask_array(self.reaction_time_mixed,self.react_switch_mask)      
        
    def calc_wins_indecisions_incorrects(self):
        self.player_wins = np.count_nonzero(self.win_mask,axis=1)
        self.player_indecisions = np.count_nonzero(self.indecision_mask,axis=1)
        self.player_incorrects = np.count_nonzero(self.incorrect_mask,axis=1)
        
        self.player_perc_wins = (self.player_wins/self.num_trials)*100
        self.player_perc_indecisions = (self.player_indecisions/self.num_trials)*100
        self.player_perc_incorrects = (self.player_incorrects/self.num_trials)*100
        
    def leave_and_reach_times_on_wins_incorrects_indecisions(self):
        win_mask = np.logical_or(self.player_task_decision_array*self.agent_task_decision_array == 1, np.logical_and(self.player_task_decision_array!=0, self.agent_task_decision_array==0))
        incorrect_mask = self.player_task_decision_array*self.agent_task_decision_array == -1
        indecision_mask = self.player_task_decision_array == 0
        # Set agent arrays
        self.agent_task_leave_time_on_wins        = self.mask_array(self.agent_task_leave_time,win_mask)
        self.agent_task_leave_time_on_indecisions = self.mask_array(self.agent_task_leave_time,indecision_mask)
        self.agent_task_leave_time_on_incorrects  = self.mask_array(self.agent_task_leave_time,incorrect_mask)
        
        # Set arrays
        self.player_task_leave_time_on_wins           = self.mask_array(self.player_task_leave_time,win_mask)
        self.player_task_leave_time_on_indecisions    = self.mask_array(self.player_task_leave_time,indecision_mask)
        self.player_task_leave_time_on_incorrects     = self.mask_array(self.player_task_leave_time,incorrect_mask)
        
        self.player_task_reach_time_on_wins           = self.mask_array(self.player_task_reach_time,win_mask)
        self.player_task_reach_time_on_indecisions    = self.mask_array(self.player_task_reach_time,indecision_mask)
        self.player_task_reach_time_on_incorrects     = self.mask_array(self.player_task_reach_time,incorrect_mask)
        
        self.player_minus_agent_leave_time_on_wins          = self.player_task_leave_time_on_wins - self.agent_task_leave_time_on_wins
        self.player_minus_agent_leave_time_on_indecisions   = self.player_task_leave_time_on_indecisions - self.agent_task_leave_time_on_indecisions
        self.player_minus_agent_leave_time_on_incorrects    = self.player_task_leave_time_on_incorrects - self.agent_task_leave_time_on_incorrects
        
    def reaction_gamble_calculations(self):
        # Gamble calculations
        if True:
            # Create mask and get the total number of gamble decisions
            self.gamble_task_mask = (self.player_task_leave_time-self.agent_task_leave_time)<=self.adjusted_player_reaction_time
            self.total_gambles = np.count_nonzero(self.gamble_task_mask,axis=1)
            
            # Get the leave time and reach target time
            self.player_gamble_task_leave_time          = self.mask_array(self.player_task_leave_time,self.gamble_task_mask)
            self.player_gamble_task_reach_time          = self.mask_array(self.player_task_reach_time,self.gamble_task_mask)
            self.agent_gamble_task_leave_time           = self.mask_array(self.agent_task_leave_time,self.gamble_task_mask)
            self.agent_gamble_task_reach_time           = self.mask_array(self.agent_task_reach_time,self.gamble_task_mask)
            self.player_minus_agent_gamble_task_leave_time = self.player_gamble_task_leave_time - self.agent_gamble_task_leave_time

            # Wins, indecisions,incorrects
            self.gamble_results_array = self.mask_array(self.trial_results,self.gamble_task_mask)
            self.gamble_wins        = np.count_nonzero(self.gamble_results_array==1,axis=1)
            self.gamble_indecisions = np.count_nonzero(self.gamble_results_array==2,axis=1)
            self.gamble_incorrects  = np.count_nonzero(self.gamble_results_array==3,axis=1)
            
            # Get percent gamble decisions
            self.perc_gamble_decisions = self.total_gambles/self.num_trials*100
            
            # Out of the gamble decisions, how many were wins, indecisions, incorrects
            self.perc_gambles_that_were_wins = np.divide(self.gamble_wins,self.total_gambles, # gamble_wins/total_gambles
                                                            out=np.zeros_like(self.gamble_wins)*np.nan,where=self.total_gambles!=0)*100
            self.perc_gambles_that_were_indecisions = np.divide(self.gamble_indecisions,self.total_gambles, # gamble_wins/total_gambles
                                                                out=np.zeros_like(self.gamble_wins)*np.nan,where=self.total_gambles!=0)*100
            self.perc_gambles_that_were_incorrects = np.divide(self.gamble_incorrects,self.total_gambles, # gamble_wins/total_gambles
                                                                out=np.zeros_like(self.gamble_wins)*np.nan,where=self.total_gambles!=0)*100
            
            # Out of the wins, indecisions, incorrecrs, how many were gambles
            self.perc_wins_that_were_gambles = np.divide(self.gamble_wins,self.player_wins, # gamble_wins/total_gambles
                                                                out=np.zeros_like(self.gamble_wins)*np.nan,where=self.player_wins!=0)*100
            self.perc_indecisions_that_were_gambles = np.divide(self.gamble_indecisions,self.player_indecisions, # gamble_indecisions/total_gambles
                                                                out=np.zeros_like(self.gamble_indecisions)*np.nan,where=self.player_indecisions!=0)*100
            self.perc_incorrects_that_were_gambles = np.divide(self.gamble_incorrects,self.player_incorrects, # gamble_incorrects/total_gambles
                                                                out=np.zeros_like(self.gamble_incorrects)*np.nan,where=self.player_incorrects!=0)*100
            
            # Get the gamble both decided metrics
            self.gamble_both_decided_mask = np.logical_and(self.gamble_task_mask,self.both_decided_mask)
            self.total_gambles_when_both_decide = np.count_nonzero(self.gamble_both_decided_mask,axis=1)
            
            self.gamble_wins_when_both_decide        = np.count_nonzero(self.mask_array(self.trial_results,self.gamble_both_decided_mask) == 1,axis=1) # Count where the both decided gamble trials are equal to 1 for the trial results array
            self.gamble_indecisions_when_both_decide = np.count_nonzero(self.mask_array(self.trial_results,self.gamble_both_decided_mask) == 2,axis=1) # Count where the both decided gamble trials are equal to 1 for the trial results array
            self.gamble_incorrects_when_both_decide  = np.count_nonzero(self.mask_array(self.trial_results,self.gamble_both_decided_mask) == 3,axis=1) # Count where the both decided gamble trials are equal to 1 for the trial results array
            
            self.perc_gamble_wins_when_both_decide       = np.divide(self.gamble_wins_when_both_decide,self.total_gambles_when_both_decide, # gamble_wins_when_both_decide/total_gambles
                                                                out=np.zeros_like(self.gamble_wins_when_both_decide)*np.nan,where=self.total_gambles_when_both_decide!=0)*100
            self.perc_gamble_incorrects_when_both_decide = np.divide(self.gamble_incorrects_when_both_decide,self.total_gambles_when_both_decide, # gamble_incorrects_when_both_decide/total_gambles
                                                                out=np.zeros_like(self.gamble_incorrects_when_both_decide)*np.nan,where=self.total_gambles_when_both_decide!=0)*100
            
        #################################
        # Reaction calculations
        if True:
            # Create mask and get the total number of reaction decisions
            self.reaction_task_mask   = (self.player_task_leave_time-self.agent_task_leave_time)>self.adjusted_player_reaction_time
            self.total_reactions = np.count_nonzero(self.reaction_task_mask,axis=1)
            
            # Get the leave time and reach target time
            self.player_reaction_task_leave_time             = self.mask_array(self.player_task_leave_time,self.reaction_task_mask)
            self.player_reaction_task_reach_time             = self.mask_array(self.player_task_reach_time,self.reaction_task_mask)
            self.agent_reaction_task_leave_time              = self.mask_array(self.agent_task_leave_time,self.reaction_task_mask)
            self.agent_reaction_task_reach_time              = self.mask_array(self.agent_task_reach_time,self.reaction_task_mask)
            self.player_minus_agent_reaction_task_leave_time = self.player_reaction_task_leave_time - self.agent_reaction_task_leave_time

            # Wins, indecisions,incorrects
            self.reaction_results_array = self.mask_array(self.trial_results,self.reaction_task_mask)
            self.reaction_wins          = np.count_nonzero(self.reaction_results_array==1,axis=1)
            self.reaction_indecisions   = np.count_nonzero(self.reaction_results_array==2,axis=1)
            self.reaction_incorrects    = np.count_nonzero(self.reaction_results_array==3,axis=1)
            
            # Get percent reaction decisions
            self.perc_reaction_decisions = self.total_reactions/self.num_trials*100
            
            # Out of the reaction decisions, how many were wins, indecisions, incorrects
            self.perc_reactions_that_were_wins        = np.divide(self.reaction_wins,self.total_reactions, # reaction_wins/total_reactions
                                                            out=np.zeros_like(self.reaction_wins)*np.nan,where=self.total_reactions!=0)*100
            self.perc_reactions_that_were_indecisions = np.divide(self.reaction_indecisions,self.total_reactions, # reaction_wins/total_reactions
                                                                out=np.zeros_like(self.reaction_wins)*np.nan,where=self.total_reactions!=0)*100
            self.perc_reactions_that_were_incorrects  = np.divide(self.reaction_incorrects,self.total_reactions, # reaction_wins/total_reactions
                                                                out=np.zeros_like(self.reaction_wins)*np.nan,where=self.total_reactions!=0)*100
            
            # Out of the wins, indecisions, incorrecrs, how many were reactions
            self.perc_wins_that_were_reactions        = np.divide(self.reaction_wins,self.player_wins, # reaction_wins/total_reactions
                                                                out=np.zeros_like(self.reaction_wins)*np.nan,where=self.player_wins!=0)*100
            self.perc_indecisions_that_were_reactions = np.divide(self.reaction_indecisions,self.player_indecisions, # reaction_indecisions/total_reactions
                                                                out=np.zeros_like(self.reaction_indecisions)*np.nan,where=self.player_indecisions!=0)*100
            self.perc_incorrects_that_were_reactions  = np.divide(self.reaction_incorrects,self.player_incorrects, # reaction_incorrects/total_reactions
                                                                out=np.zeros_like(self.reaction_incorrects)*np.nan,where=self.player_incorrects!=0)*100
            
            # Get the reaction both decided metrics
            self.reaction_both_decided_mask       = np.logical_and(self.reaction_task_mask,self.both_decided_mask)
            self.total_reactions_when_both_decide = np.count_nonzero(self.reaction_both_decided_mask,axis=1)
            
            self.reaction_wins_when_both_decide        = np.count_nonzero(self.mask_array(self.trial_results,self.reaction_both_decided_mask) == 1,axis=1) # Count where the both decided reaction trials are equal to 1 for the trial results array
            self.reaction_indecisions_when_both_decide = np.count_nonzero(self.mask_array(self.trial_results,self.reaction_both_decided_mask) == 2,axis=1) # Count where the both decided reaction trials are equal to 1 for the trial results array
            self.reaction_incorrects_when_both_decide  = np.count_nonzero(self.mask_array(self.trial_results,self.reaction_both_decided_mask) == 3,axis=1) # Count where the both decided reaction trials are equal to 1 for the trial results array
            
            self.perc_reaction_wins_when_both_decide       = np.divide(self.reaction_wins_when_both_decide,self.total_reactions_when_both_decide, # reaction_wins_when_both_decide/total_reactions
                                                                out=np.zeros_like(self.reaction_wins_when_both_decide)*np.nan,where=self.total_reactions_when_both_decide!=0)*100
            self.perc_reaction_incorrects_when_both_decide = np.divide(self.reaction_incorrects_when_both_decide,self.total_reactions_when_both_decide, # reaction_incorrects_when_both_decide/total_reactions
                                                                out=np.zeros_like(self.reaction_incorrects_when_both_decide)*np.nan,where=self.total_reactions_when_both_decide!=0)*100    
        
    def wins_when_both_decide(self):
        # Get agent leave array
        self.player_perc_both_reached_wins = np.zeros((self.num_blocks))
        self.agent_both_reached_wins = np.zeros((self.num_blocks))
        # Get wins when both decide
        win_mask = self.agent_task_decision_array*self.player_task_decision_array==1
        incorrect_mask = self.agent_task_decision_array*self.player_task_decision_array==-1
        checker = self.player_task_decision_array != 0 
        
        total_both_reached_mask = np.logical_and(self.player_task_decision_array!=0, self.agent_task_decision_array!=0)
        
        self.player_both_reached_wins = np.count_nonzero(win_mask,axis=1)        
        self.player_both_reached_incorrects = np.count_nonzero(incorrect_mask,axis=1)   
        self.total_both_reached = np.count_nonzero(total_both_reached_mask,axis=1)

        self.player_perc_both_reached_wins = (self.player_both_reached_wins/self.total_both_reached)*100
        self.player_perc_both_reached_incorrects = (self.player_both_reached_incorrects/self.total_both_reached)*100
        
    def predict_decision_times(self,gamble_delay):
        reaction_mask = self.player_task_leave_time - self.agent_task_leave_time>=self.adjusted_player_reaction_time
        gamble_mask = self.player_task_leave_time - self.agent_task_leave_time<self.adjusted_player_reaction_time
        self.player_predicted_decision_time = (self.perc_reaction_decisions/100)*(self.agent_reaction_leave_time_mean) + \
                                               (self.perc_gamble_decisions/100)*(self.player_gamble_leave_time_mean - gamble_delay)  
          
             
    def binned_metrics(self,bin_start = 800,bin_end = 1400, bin_size = 50,cut_off_threshold = 30):
        self.bins = np.arange(bin_start,bin_end,bin_size)
        self.bin_length_each_condition                    = np.zeros((len(self.bins)-1,self.num_blocks))
        self.binned_player_task_leave_times               = np.zeros((len(self.bins)-1,self.num_blocks,self.num_trials))*np.nan 
        self.binned_player_task_decision_array            = np.zeros((len(self.bins)-1,self.num_blocks,self.num_trials))*np.nan 
        self.binned_agent_task_leave_times                = np.zeros((len(self.bins)-1,self.num_blocks,self.num_trials))*np.nan 
        self.binned_player_minus_agent_task_leave_time    = np.zeros((len(self.bins)-1,self.num_blocks,self.num_trials))*np.nan 
        self.binned_agent_task_decision_array             = np.zeros((len(self.bins)-1,self.num_blocks,self.num_trials))*np.nan 
        self.binned_player_wins                           = np.zeros((len(self.bins)-1,self.num_blocks))
        self.binned_player_indecisions                    = np.zeros((len(self.bins)-1,self.num_blocks))
        self.binned_player_incorrects                     = np.zeros((len(self.bins)-1,self.num_blocks))
        self.mean_binned_player_wins                      = np.zeros((len(self.bins)-1,self.num_blocks))
        self.mean_binned_player_indecisions               = np.zeros((len(self.bins)-1,self.num_blocks))
        self.mean_binned_player_incorrects                = np.zeros((len(self.bins)-1,self.num_blocks))
        for b in range(len(self.bins)-1):
            bin_index = np.argwhere((self.bins[b] < self.agent_task_leave_time) & (self.agent_task_leave_time < self.bins[b+1]))
            for i,j in bin_index:
                self.bin_length_each_condition[b,i]+=1
                self.binned_player_task_leave_times[b,i,j]    = self.player_task_leave_time[i,j] 
                self.binned_player_task_decision_array[b,i,j] = self.player_task_decision_array[i,j]
                self.binned_agent_task_leave_times[b,i,j]     = self.agent_task_leave_time[i,j]
                self.binned_agent_task_decision_array[b,i,j]  = self.agent_task_decision_array[i,j] 
                if ((self.player_task_decision_array[i,j]*self.agent_task_decision_array[i,j] == 1) or (self.player_task_decision_array[i,j] != 0 and self.agent_task_decision_array[i,j] == 0)):
                    self.binned_player_wins[b,i] += 1
                elif self.player_task_decision_array[i,j] == 0:
                    self.binned_player_indecisions[b,i] += 1
                elif (self.player_task_decision_array[i,j]*self.agent_task_decision_array[i,j] == -1):
                    self.binned_player_incorrects[b,i] += 1        
                
        self.binned_player_minus_agent_task_leave_time = self.binned_player_task_leave_times - self.binned_agent_task_leave_times
        # Get percentages based on bin length    
         
        self.perc_binned_player_wins        =  np.divide(self.binned_player_wins,self.bin_length_each_condition,
                                                     out=np.zeros_like(self.binned_player_wins)*np.nan,where=self.bin_length_each_condition!=0)*100
        self.perc_binned_player_indecisions =  np.divide(self.binned_player_indecisions,self.bin_length_each_condition,
                                                     out=np.zeros_like(self.binned_player_indecisions)*np.nan,where=self.bin_length_each_condition!=0)*100
        self.perc_binned_player_incorrects  =  np.divide(self.binned_player_incorrects,self.bin_length_each_condition,
                                                     out=np.zeros_like(self.binned_player_incorrects)*np.nan,where=self.bin_length_each_condition!=0)*100
        # Calculate mean across all trials
        self.binned_player_task_leave_times_mean            = np.nanmean(self.binned_player_task_leave_times,axis=2) # Mean for each bin, each condition
        self.binned_player_minus_agent_task_leave_time_mean = np.nanmean(self.binned_player_minus_agent_task_leave_time,axis=2)
        
        # Cut off at threshold
        mask = self.bin_length_each_condition>cut_off_threshold
        self.perc_binned_player_wins_cutoff                           = self.perc_binned_player_wins*mask
        self.perc_binned_player_indecisions_cutoff                    = self.perc_binned_player_indecisions*mask
        self.perc_binned_player_incorrects_cutoff                     = self.perc_binned_player_incorrects*mask
        self.binned_player_minus_agent_task_leave_time_mean_cutoff = self.binned_player_minus_agent_task_leave_time_mean*mask
        self.binned_player_task_leave_times_mean_cutoff            = self.binned_player_task_leave_times_mean*mask

class Group():
    def __init__(self, objects,**kwargs):
        self.objects = objects
        self.num_blocks = kwargs.get('num_blocks',4)
        self.num_trials = kwargs.get('num_trials',80)
        self.bin_cutoff_threshold = kwargs.get('bin_cutoff_threshold',30)
        
    def analyze_data(self):
        #* Analyze data function for each object 
        for i,o in enumerate(self.objects):
            o.analyze_data()
            self.objects[i] = o
            
        #* Flatten the trial by trial data so I can make a histogram including everyones data
        self.flatten_across_all_subjects()

        #* Loop through all attributes, and set the group attribute with all subjects combined
        for a in dir(self.objects[0]):
            if not a.startswith('__'):
                setattr(self,a,np.array([getattr(o,a) for o in self.objects]))
                
    def flatten_across_all_subjects(self):
        self.all_player_task_leave_times_each_condition            = self.concatenate_across_subjects('player_task_leave_time')
        self.all_player_task_gamble_leave_times_each_condition     = self.concatenate_across_subjects('player_gamble_task_leave_time')
        self.all_player_task_reaction_leave_times_each_condition   = self.concatenate_across_subjects('player_reaction_task_leave_time')
    
        self.all_agent_task_leave_times_each_condition            = self.concatenate_across_subjects('agent_task_leave_time')
        self.all_agent_task_gamble_leave_times_each_condition     = self.concatenate_across_subjects('agent_gamble_task_leave_time')
        self.all_agent_task_reaction_leave_times_each_condition   = self.concatenate_across_subjects('agent_reaction_task_leave_time')
        
    
    def binning_across_all_subjects(self):  
        # BINNING ----------------------------------------------------------------------------------------
        # Binned mean across all participants
        self.perc_binned_player_wins_mean                      = np.nanmean(self.combine_all_subjects('perc_binned_player_wins'),axis = 0)
        self.perc_binned_player_indecisions_mean               = np.nanmean(self.combine_all_subjects('perc_binned_player_indecisions'),axis = 0)
        self.perc_binned_player_incorrects_mean                = np.nanmean(self.combine_all_subjects('perc_binned_player_incorrects'),axis = 0)
        self.binned_player_minus_agent_task_leave_time_mean    = np.nanmean(self.combine_all_subjects('binned_player_minus_agent_task_leave_time_mean'),axis = 0)
        self.binned_player_task_leave_times_mean               = np.nanmean(self.combine_all_subjects('binned_player_task_leave_times_mean'),axis = 0)
        
        # Combine subjects into array
        self.bin_length_each_subject_each_condition       = self.combine_all_subjects('bin_length_each_condition')
        self.bin_length_each_condition                    = np.sum(self.bin_length_each_subject_each_condition,axis=0)
        self.binned_player_minus_agent_task_leave_time    = self.combine_all_subjects('binned_player_minus_agent_task_leave_time')
        self.perc_binned_player_wins                      = self.combine_all_subjects('perc_binned_player_wins')
        self.perc_binned_player_indecisions               = self.combine_all_subjects('perc_binned_player_indecisions')
        self.perc_binned_player_incorrects                = self.combine_all_subjects('perc_binned_player_incorrects')
        
        self.bin_threshold()
        
        
    def combine_all_subjects(self,metric):
        '''
        List comprehension into np array to put the subjects at index 0
        '''
        return np.array([getattr(o,metric) for o in self.objects])
    
    def first_half_second_half(self,metric):
        arr = np.array([getattr(o,metric) for o in self.objects])
        
    def concatenate_across_subjects(self,metric):
        '''
        Flattens out the subject dimension to get array of all the subjects 
        
        Usually used for group distributions
        '''
        arr = self.combine_all_subjects(metric)
        temp = np.swapaxes(arr,0,1)
        ans = np.reshape(temp,(self.num_blocks,-1))
        return ans
    
    def predict_decision_times(self,gamble_delay,weird_delay):    
        self.player_predicted_decision_time = np.array([(o.perc_reaction_decisions/100)*(o.agent_reaction_leave_time_mean) + (o.perc_gamble_decisions/100)*(o.player_gamble_leave_time_mean - gamble_delay) for o in self.objects]) 
        self.predict_stopping_time_from_reactions_gambles(weird_delay=weird_delay)
    
    def predict_stopping_time_from_reactions_gambles(self,weird_delay):
        '''
        Using the percentage reactions and gambles to predict the stopping time
        '''
        timesteps = np.arange(500,1500,1)
        self.predicted_stopping_time = np.zeros((len(self.objects),6))
        self.predicted_stopping_time_index = np.zeros((len(self.objects),6))
        self.predicted_perc_reaction_decisions = np.zeros((len(self.objects),6,len(timesteps)))
        self.predicted_perc_gamble_decisions = np.zeros((len(self.objects),6,len(timesteps)))
        self.final_predicted_perc_reaction_decisions = np.zeros((len(self.objects),6))
        self.final_predicted_perc_gamble_decisions = np.zeros((len(self.objects),6))
        react_loss = np.zeros((len(self.objects),6,len(timesteps)))
        gamble_loss = np.zeros((len(self.objects),6,len(timesteps)))
        for i,o in enumerate(self.objects):
            for j in range(6):
                for k,t in enumerate(timesteps):
                    self.predicted_perc_reaction_decisions[i,j,k] = np.count_nonzero(o.agent_task_leave_time[j,:]<=t+weird_delay)/o.num_trials*100
                    react_loss[i,j,k] = abs(o.perc_reaction_decisions[j] - self.predicted_perc_reaction_decisions[i,j,k])
                    self.predicted_perc_gamble_decisions[i,j,k] = np.count_nonzero(o.agent_task_leave_time[j,:]>t+weird_delay)/o.num_trials*100
                    gamble_loss[i,j,k] = abs(o.perc_gamble_decisions[j] - self.predicted_perc_gamble_decisions[i,j,k])
                self.predicted_stopping_time_index[i,j] = np.argmin(react_loss[i,j,:]+gamble_loss[i,j,:])
                self.final_predicted_perc_reaction_decisions[i,j] = self.predicted_perc_reaction_decisions[i,j,int(self.predicted_stopping_time_index[i,j])]
                self.final_predicted_perc_gamble_decisions[i,j]   = self.predicted_perc_gamble_decisions[i,j,int(self.predicted_stopping_time_index[i,j])]
        self.predicted_stopping_time = self.predicted_stopping_time_index + np.min(timesteps)
        
    def find_subject(self,metric,comparison_num,comparison_direction):
        '''
        Used to find the subject who's specific value is greater or less than the inputted comparison metric
        '''
        metrics = self.combine_all_subjects(metric)
        for i,m in enumerate(metrics):
            if comparison_direction == 'greater than':
                if m.any() > comparison_num:
                    print(f'Sub{i+1}')
            if comparison_direction == 'less than':
                if m.any() < comparison_num:
                    print(f'Sub{i+1}')
    
    def filter_gamble_reaction_under_10(self,metric):
        for o in self.objects:
            for i in range(self.num_blocks):
                if o.total_reactions[i]<10:
                    o.perc_reactions_that_were_wins[i] = np.nan
                    o.perc_reactions_that_were_indecisions[i] = np.nan
                    o.perc_reactions_that_were_incorrects[i] = np.nan
                    o.player_reaction_leave_time_mean[i] = np.nan
                    o.player_reaction_leave_time_sd[i] = np.nan
                    o.perc_reaction_wins_when_both_decide[i] = np.nan
                    o.perc_reaction_incorrects_when_both_decide[i] = np.nan

                if o.total_gambles[i]<10:
                    o.perc_gambles_that_were_wins[i] = np.nan
                    o.perc_gambles_that_were_incorrects[i] = np.nan
                    o.perc_gambles_that_were_indecisions[i] = np.nan
                    o.player_gamble_leave_time_mean[i] = np.nan
                    o.player_gamble_leave_time_sd[i] = np.nan
                    o.perc_gamble_wins_when_both_decide[i] = np.nan
                    o.perc_gamble_incorrects_when_both_decide[i] = np.nan
                         
    def bin_threshold(self):
        '''
        Cut off bins so they only show if the bin has over the amount of the certain threshold
        '''
        # Cut off at threshold
        mask = self.bin_length_each_condition>self.bin_cutoff_threshold
        self.perc_binned_player_wins_mean_cutoff                      = self.perc_binned_player_wins_mean*mask
        self.perc_binned_player_indecisions_mean_cutoff               = self.perc_binned_player_indecisions_mean*mask
        self.perc_binned_player_incorrects_mean_cutoff                = self.perc_binned_player_incorrects_mean*mask
        self.binned_player_minus_agent_task_leave_time_mean_cutoff = self.binned_player_minus_agent_task_leave_time_mean*mask
        self.binned_player_task_leave_times_mean_cutoff            = self.binned_player_task_leave_times_mean*mask
        