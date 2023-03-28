import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
class Subject():
    def __init__(self,**kwargs):
        self.num_blocks                             = kwargs.get('num_blocks')
        self.num_trials                             = kwargs.get('num_trials')
        self.num_trials_50                          = int(self.num_trials/2)
        self.trial_type_array                       = kwargs.get('trial_type_array')
        self.reaction_times                         = kwargs.get('reaction_times')
        
        self.gamble_mask                 = self.trial_type_array == 0
        self.react_mask                  = self.trial_type_array == 1
        self.gamble_reaction_time_all    = self.mask_array(self.reaction_times,self.gamble_mask) 
        self.react_reaction_time_all     = self.mask_array(self.reaction_times,self.react_mask) 
        
        self.reaction_times_mixed = self.reaction_times[0,:]
        
        self.gamble_reaction_time_only_gamble        = self.gamble_reaction_time_all[2,:]
        self.gamble_reaction_time_mixed = self.gamble_reaction_time_all[0,:]
        
        self.react_reaction_time_only_react         = self.react_reaction_time_all[1,:]
        self.react_reaction_time_mixed  = self.react_reaction_time_all[0,:]
        
        self.calculate_means_and_sds()
        self.calculate_repeats_alternates()
        self.remove_nans()
    def calculate_means_and_sds(self):
        self.gamble_reaction_time_only_gamble_mean          = np.nanmean(self.gamble_reaction_time_only_gamble)
        self.gamble_reaction_time_only_gamble_median        = np.nanmedian(self.gamble_reaction_time_only_gamble)
        self.gamble_reaction_time_only_gamble_sd            = np.nanstd(self.gamble_reaction_time_only_gamble)
        self.gamble_reaction_time_mixed_mean   = np.nanmean(self.gamble_reaction_time_mixed)
        self.gamble_reaction_time_mixed_median = np.nanmedian(self.gamble_reaction_time_mixed)
        self.gamble_reaction_time_mixed_sd     = np.nanstd(self.gamble_reaction_time_mixed)
        
        self.react_reaction_time_only_react_mean          = np.nanmean(self.react_reaction_time_only_react)
        self.react_reaction_time_only_react_median        = np.nanmedian(self.react_reaction_time_only_react)
        self.react_reaction_time_only_react_sd            = np.nanstd(self.react_reaction_time_only_react)
        self.react_reaction_time_mixed_mean   = np.nanmean(self.react_reaction_time_mixed)
        self.react_reaction_time_mixed_median = np.nanmedian(self.react_reaction_time_mixed)
        self.react_reaction_time_mixed_sd     = np.nanstd(self.react_reaction_time_mixed)
    def calculate_repeats_alternates(self):
        # Get masks
        self.react_repeat_mask  = np.full(self.num_trials,False)
        self.react_switch_mask  = np.full(self.num_trials,False)
        self.gamble_repeat_mask = np.full(self.num_trials,False)
        self.gamble_switch_mask = np.full(self.num_trials,False)
        for i in range(self.num_trials-1):
            if (self.trial_type_array[0,i]==1 and self.trial_type_array[0,i+1] == 1):
                self.react_repeat_mask[i+1]  = True
                self.react_switch_mask[i+1]  = False
                self.gamble_repeat_mask[i+1] = False
                self.gamble_switch_mask[i+1] = False
            elif (self.trial_type_array[0,i]==1 and self.trial_type_array[0,i+1]==0):
                self.react_repeat_mask[i+1]  = False
                self.react_switch_mask[i+1]  = False
                self.gamble_repeat_mask[i+1] = False
                self.gamble_switch_mask[i+1] = True   
            elif (self.trial_type_array[0,i]==0 and self.trial_type_array[0,i+1]==1):
                self.react_repeat_mask[i+1]  = False
                self.react_switch_mask[i+1]  = True
                self.gamble_repeat_mask[i+1] = False
                self.gamble_switch_mask[i+1] = False
            elif (self.trial_type_array[0,i]==0 and self.trial_type_array[0,i+1]==0):
                self.react_repeat_mask[i+1]  = False
                self.react_switch_mask[i+1]  = False
                self.gamble_repeat_mask[i+1] = True
                self.gamble_switch_mask[i+1] = False
        
        # Get the reaction times of gambles on repeats and switchs
        self.gamble_reaction_time_repeat = self.mask_array(self.reaction_times_mixed,self.gamble_repeat_mask)
        self.gamble_reaction_time_switch = self.mask_array(self.reaction_times_mixed,self.gamble_switch_mask)
        # Get the reaction times of reactions on repeats and mixeds
        self.react_reaction_time_repeat  = self.mask_array(self.reaction_times_mixed,self.react_repeat_mask)
        self.react_reaction_time_switch  = self.mask_array(self.reaction_times_mixed,self.react_switch_mask)
        
    def mask_array(self,arr,mask):
            '''
            Applies the mask to the array then replaces the 0s with nans
            '''
            new_arr = arr*mask # Apply mask
            new_arr[~mask] = np.nan # Replace the 0s from the mask with np nan
            return new_arr
    def remove_nans(self,arr=None,mask=None):
        gamble_nanmask = np.isnan(self.gamble_reaction_time_all)
        self.gamble_reaction_time_only_gamble        = self.gamble_reaction_time_all[2,:][~gamble_nanmask[2,:]]
        self.gamble_reaction_time_mixed              = self.gamble_reaction_time_all[0,:][~gamble_nanmask[0,:]]
        
        react_nanmask = np.isnan(self.react_reaction_time_all)
        self.react_reaction_time_only_react         = self.react_reaction_time_all[1,:][~react_nanmask[1,:]]
        self.react_reaction_time_mixed              = self.react_reaction_time_all[0,:][~react_nanmask[0,:]]
        
class Group():
    def __init__(self, objects,**kwargs):
        self.objects = objects
        self.num_blocks = kwargs.get('num_blocks',6)
        self.num_trials = kwargs.get('num_trials',80)
        self.bin_cutoff_threshold = kwargs.get('bin_cutoff_threshold',30)
        
    def analyze_data(self):
        # self.gamble_reaction_time_mean       = np.nanmean(self.combine_all_subjects('gamble_reaction_time_mean'))
        # self.gamble_reaction_time_median     = np.nanmedian(self.combine_all_subjects('gamble_reaction_time_mean'))
        # self.gamble_reaction_time_sd         = np.nanmean(self.combine_all_subjects('gamble_reaction_time_sd'))
        return
    def combine_all_subjects(self,metric):
        '''
        List comprehension into np array to put the subjects at index 0
        '''
        return np.array([getattr(o,metric) for o in self.objects])
    
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
                    
    def concatenate_across_subjects(self,metric):
        '''
        Flattens out the subject dimension to get array of all the subjects 
        
        Usually used for group distributions
        '''
        arr = self.combine_all_subjects(metric)
        temp = np.swapaxes(arr,0,1)
        ans = np.reshape(temp,(self.num_blocks,-1))
        return ans