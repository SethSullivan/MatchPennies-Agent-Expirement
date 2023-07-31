import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import warnings

def mask_array(self,arr,mask):
    '''
    Applies the mask to the array then replaces the 0s with nans
    '''
    new_arr = arr*mask # Apply mask
    new_arr[~mask] = np.nan # Replace the 0s from the mask with np nan
    return new_arr

class TargetInfo:
    def __init__(self, **kwargs):
        #* Target information is same for both Exp1 and Exp2 for Aim 1, so just use this file for both
        self.filename = kwargs.get('filename', 'D:\OneDrive - University of Delaware - o365\Subject_Data\MatchPennies_Agent_Exp2\\Sub1_Task\\Sub1_TaskTarget_Table.csv')
        df = pd.read_csv(self.filename)
        df["X"] = df["X"]/100
        df["Y"] = df["Y"]/100
        df['Dim 1'] = df['Dim 1']/100 # Target table is in centimeters, I guess this doesn't matter but it makes me feel better
        df['Dim 2'] = df['Dim 2']/100
        # Target information for Right Hand (keeping this because the positions of target 3 and 4 are based on target 1 and start 1)
        self.startx         = df.loc[0]['X']
        self.starty         = df.loc[0]['Y']
        self.start_radius   = df.loc[0]['Dim 1'] 
        self.target1x       = df.loc[1]['X']
        self.target1y       = df.loc[1]['Y']
        self.target1_radius = df.loc[1]['Dim 1']
        self.target2x       = 2*self.startx - self.target1x
        self.target2y       = self.target1y
        self.target2_radius = self.target1_radius
        # Timing target
        self.timing_targetx       = self.startx
        self.timing_targety       = self.target1y
        self.timing_target_radius = self.target1_radius 

class ExperimentInfo:
    def __init__(self,subject,experiment,num_task_blocks,num_task_trials_initial,num_reaction_blocks,
                 num_reaction_trials,num_timing_trials,select_trials):
        #* Get Experimental Design Data
            self.subject                      = subject
            self.experiment                   = experiment
            self.num_task_blocks              = num_task_blocks
            self.num_task_trials_initial      = num_task_trials_initial
            self.num_reaction_blocks          = num_reaction_blocks
            self.num_reaction_trials          = num_reaction_trials
            self.num_timing_trials            = num_timing_trials
            self.select_trials                = select_trials
            if self.select_trials != 'All Trials':
                self.num_task_trials = self.num_task_trials_initial//2
            else:
                self.num_task_trials = self.num_task_trials_initial
            # Make sure I don't have the wrong experiment in there
            if self.experiment == 'Exp2':
                assert self.num_task_blocks == 4 
                
    def __repr__(self) -> str:
        return f'Subject: {self.subject}, Experiment: {self.experiment} {self.__class__.__name__} Object'

class RawData():
    def __init__(self, exp_info: ExperimentInfo, 
                 reaction_xypos_data,reaction_dist_data, reaction_xyvelocity_data, 
                 reaction_speed_data, reaction_trial_type_array, reaction_trial_start, 
                 agent_reaction_decision_array, agent_reaction_leave_time, interval_trial_start, 
                 interval_reach_time, coincidence_trial_start, coincidence_reach_time, 
                 task_xypos_data, task_dist_data, task_xyvelocity_data, task_speed_data, 
                 agent_task_decision_array, agent_task_leave_time,
                 ):
        self.exp_info = exp_info
        #* Reaction Data
        if True:
            self.reaction_xypos_data            = reaction_xypos_data
            self.reaction_dist_data             = reaction_dist_data
            self.reaction_xyvelocity_data       = reaction_xyvelocity_data
            self.reaction_speed_data            = reaction_speed_data
            # self.reaction_xyforce_data          = reaction_xyforce_data
            # self.reaction_force_data            = reaction_force_data
            self.reaction_trial_type_array      = reaction_trial_type_array
            self.reaction_trial_start           = reaction_trial_start
            self.agent_reaction_decision_array  = agent_reaction_decision_array
            self.agent_reaction_leave_time      = agent_reaction_leave_time
            self.agent_reaction_reach_time      = self.agent_reaction_leave_time + 150
            self.agent_reaction_decision_array[self.agent_reaction_reach_time>1500] = 0
        #* Timing Data
        if True:
            self.interval_trial_start           = interval_trial_start
            self.interval_reach_time            = interval_reach_time
            self.coincidence_trial_start        = coincidence_trial_start
            self.coincidence_reach_time         = coincidence_reach_time
        #* Task data
        if True:
            self.task_xypos_data                = self.slice_array(task_xypos_data)
            self.task_dist_data                 = self.slice_array(task_dist_data)
            self.task_xyvelocity_data           = self.slice_array(task_xyvelocity_data)
            self.task_speed_data                = self.slice_array(task_speed_data)
            # self.task_xyforce_data              = self.slice_array(task_xyforce_data)
            # self.task_force_data                = self.slice_array(task_force_data)
            self.agent_task_decision_array      = self.slice_array(agent_task_decision_array)
            self.agent_task_leave_time          = self.slice_array(agent_task_leave_time)
            self.agent_task_reach_time          = self.agent_task_leave_time + 150
            
            self.agent_task_decision_array[self.agent_task_reach_time>1500] = 0
                    
        # self.player_task_leave_time                = self.slice_array(kwargs.get('player_task_leave_time'))
        # self.player_yforce_task_leave_time         = self.slice_array(kwargs.get('player_yforce_task_leave_time'))
        # self.player_task_movement_time             = self.slice_array(kwargs.get('player_task_movement_time'))
        # self.player_yforce_task_movement_time      = self.slice_array(kwargs.get('player_yforce_task_movement_time'))
        # self.player_task_reach_time                = self.slice_array(kwargs.get('player_task_reach_time'))
        # self.agent_task_leave_time                 = self.slice_array(kwargs.get('agent_task_leave_time'))
        # self.agent_task_movement_time              = self.slice_array(kwargs.get('agent_task_movement_time'))
        # self.agent_task_reach_time                 = self.slice_array(kwargs.get('agent_task_reach_time'))
        # self.player_minus_agent_task_leave_time = self.player_task_leave_time - self.agent_task_leave_time
        # self.player_yforce_minus_agent_task_leave_time = self.player_yforce_task_leave_time - self.agent_task_leave_time
        
        
        # self.reaction_time_700_mask = self.reaction_time < 700
        # self.reaction_time = self.mask_array(self.reaction_time,self.reaction_time_700_mask)
        
    def slice_array(self,arr):
        '''
        This function slices the raw data so we can do first half or second half
        '''
        if isinstance(arr,np.ndarray):
            if self.exp_info.select_trials == 'First Half':
                return arr[:,:self.exp_info.num_task_trials,...]
            elif self.exp_info.select_trials == 'Second Half':
                return arr[:,self.exp_info.num_task_trials:,...]
            elif self.exp_info.select_trials == 'All Trials':
                return arr
            else:
                raise ValueError('exp_info.select_trials should be First Half, Second Half, or All Trials')
        else:
            return arr

        
        
class MovementMetrics:
    def __init__(self, exp_info: ExperimentInfo, raw_data: RawData,
                 **kwargs):        
        self.exp_info = exp_info
        self.raw_data = raw_data
        self.vel_threshold = kwargs.get('velocity_threshold',0.05)
        self.metric_type   = kwargs.get('metric_type','velocity')

        if self.metric_type not in ['position','velocity','velocity linear']:
            raise ValueError('type should be \'position\', \'velocity\', or \'velocity linear\'')
        
        if self.exp_info.experiment == 'Exp2':
            self.reaction_gamble_mask = self.raw_data.reaction_trial_type_array == 0
            self.reaction_react_mask = self.raw_data.reaction_trial_type_array == 1
        
        self.big_num = 100000
    
    def right_left_target_ids(self,task):
        if task == 'reaction':
            xydata = self.raw_data.reaction_xypos_data
        elif task == 'task':
            xydata = self.raw_data.task_xypos_data
        else:
            raise ValueError('task should be \'task\' or \'reaction\'')
        # Argmax finds the first id where the condition is true
        enter_right_target_id = np.argmax(np.sqrt((xydata[...,0]-self.target1x)**2 + 
                                    (xydata[...,1]-self.target1y)**2) < self.target1_radius,axis=2) # Find when people enter the right target
        enter_left_target_id = np.argmax(np.sqrt((xydata[...,0]-self.target2x)**2 + 
                                    (xydata[...,1]-self.target2y)**2) < self.target2_radius,axis=2)
        # DOING THIS WAY instaed of doing np.maximum, bc sometimes people end up reahcing BOTH targets, so np.argmax returns a non-zero value (NOT RELEVANT FOR REACTION)
        # Therefore, i can't take the maximum in that case, so I need to make 0 equal 100000 so I can then take the minimum
        enter_right_target_id[enter_right_target_id == 0] = self.big_num
        enter_left_target_id[enter_left_target_id == 0] = self.big_num
        
        return enter_right_target_id, enter_left_target_id
            
    def target_reach_times(self,task):
        '''
        Calculate when the participants hand position enters the right or left target
        '''
        
        enter_right_target_id, enter_left_target_id = self.right_left_target_ids(task=task)
        
        
        if task == 'reaction':
            if self.exp_info.experiment == 'Exp1':
                enter_right_target_id = enter_right_target_id[2]
                enter_left_target_id = enter_left_target_id[2]
                
        reach_times = np.minimum(enter_right_target_id, enter_left_target_id).astype(float)
        
        reach_times[reach_times==self.big_num] = np.nan
        
        return reach_times
    
    @property
    def player_task_decision_array(self):
        #* Determine the decision array based on target selection or indecision
        enter_right_target_id, enter_left_target_id = self.right_left_target_ids(task='task')

        player_task_decision_array = np.zeros((self.exp_info.num_task_blocks,self.exp_info.num_task_trials))*np.nan
        
        player_task_decision_array[enter_right_target_id < enter_left_target_id] = 1 # Player selected right target
        player_task_decision_array[enter_left_target_id < enter_right_target_id] = -1
        player_task_decision_array[self.target_reach_times(task='task') > 1500] = 0
        player_task_decision_array[self.target_reach_times(task='task')==self.big_num] = 0
        
        return player_task_decision_array
        
    
    def movement_onset_times(self, task):
        if task == 'reaction':
            if self.metric_type == 'position':
                movement_onset_times = np.argmax(self.raw_data.reaction_dist_data > self.exp_info.start_radius,axis=2)
            elif self.metric_type == 'velocity':
                movement_onset_times = np.argmax(self.raw_data.reaction_speed_data > self.vel_threshold, axis=2)
            elif self.metric_type == 'velocity_linear':
                raise NotImplementedError('Still haven\'t implemented this in the refactor')
        elif task == 'task':
            if self.metric_type == 'position':
                movement_onset_times = np.argmax(self.raw_data.task_dist_data > self.exp_info.start_radius,axis=2)
            elif self.metric_type == 'velocity':
                movement_onset_times = np.argmax(self.raw_data.task_speed_data > self.vel_threshold,axis=2)
            elif self.metric_type == 'velocity_linear':
                raise NotImplementedError('Still haven\'t implemented this in the refactor')
        movement_onset_times = movement_onset_times.astype(np.float)
        #* Any time they never left the start should be nan, not zero
        movement_onset_times[movement_onset_times==0] = np.nan
        return movement_onset_times
        
    def movement_times(self, task) -> np.array:
        movement_time = (
            self.target_reach_times(task=task)
            - self.movement_onset_times(task=task)
        )
        
        return movement_time
    
    def player_minus_agent_movement_onset_times(self, task) -> np.array:
        return self.movement_onset_times(task=task) - self.raw_data.agent_task_movement_onset_time
    
    @property
    def reaction_times(self):
        '''
        In exp1, the start of the trial is the stimulus, so movement onset == reaction time
        In exp2, the agent is the stimulus, so the (movement onset - agent onset) == reaction time
        '''
        
        if self.exp_info.experiment == 'Exp1':
            ans = self.movement_onset_times(task='reaction')[2] #! Last row is the actual reaction, first two are timing for exp1
        elif self.exp_info.experiment == 'Exp2':
            ans = self.movement_onset_times(task='reaction') - self.raw_data.agent_reaction_leave_time
        #* Filter out fault reaction times
        ans[(ans>600)|(ans<170)] = np.nan
        return ans

    def exp2_react_gamble_reaction_time_all(self, react_or_guess):
        if react_or_guess == 'react':
            return self.reaction_times[self.reaction_react_mask].reshape(2,50)
        elif react_or_guess == 'guess':
            return self.reaction_times[self.reaction_gamble_mask].reshape(2,50)
        
    def exp2_react_gamble_reaction_time_split(self, react_or_guess, mixed_or_only):
        '''
        Only: React or Gamble reaction times during the blocks where it was only 
        the agent moving, or only it disappearing
        
        Mixed: React or Gamble reaction times during the block where the agent could either 
        move or disappear randomly
        '''
        if mixed_or_only == 'mixed':
            slice_num = 0
        elif mixed_or_only == 'only':
            slice_num = 1
            
        return self.exp2_react_gamble_reaction_time_all(react_or_guess)[slice_num,:]
        
    def exp2_react_gamble_movement_time_all(self, react_or_guess):
        '''
        All the react or gamble movement times in one (2,50) array
        '''
        if react_or_guess == 'react':
            return self.movement_times[self.reaction_react_mask].reshape(2,50)
        elif react_or_guess == 'guess':
            return self.movement_times[self.reaction_gamble_mask].reshape(2,50)
        
    def exp2_react_gamble_movement_time_split(self, react_or_guess, mixed_or_only):
        '''
        Only: React or Gamble reaction times during the blocks where it was only 
        the agent moving, or only it disappearing
        
        Mixed: React or Gamble reaction times during the block where the agent could either 
        move or disappear randomly
        '''
        if mixed_or_only == 'mixed':
            slice_num = 0
        elif mixed_or_only == 'only':
            slice_num = 1
            
        return self.exp2_react_gamble_movement_time_all(react_or_guess)[slice_num,:]
    
    def exp2_react_gamble_repeats_or_alternates(self, react_or_guess, repeats_or_alternates):
        raise NotImplementedError('Still need to implement this')

    @property
    def both_reached_mask(self):
        ans = np.logical_and(self.player_task_decision_array!=0, self.raw_data.agent_task_decision_array!=0)
        return ans

# class PlayerTargetDecisions:
#     def __init__(self, exp_info: ExperimentInfo, raw_data: RawData, movement_metrics: MovementMetrics):
#         self.exp_info = exp_info
#         self.raw_data = raw_data
#         self.movement_metrics = movement_metrics
        
#     def player_task_decision_array(self):
        
        
class ScoreMetrics:
    def __init__(self, exp_info: ExperimentInfo, raw_data: RawData, movement_metrics: MovementMetrics):
        self.exp_info = exp_info
        self.raw_data = raw_data
        self.movement_metrics = movement_metrics
        
    @property
    def win_mask(self):
        ans = np.logical_or(
            self.movement_metrics.player_task_decision_array*self.raw_data.agent_task_decision_array == 1, 
            np.logical_and(self.movement_metrics.player_task_decision_array!=0, self.raw_data.agent_task_decision_array==0)
        )
        return ans
    
    @property
    def indecision_mask(self):
        return self.movement_metrics.player_task_decision_array == 0
    
    @property
    def incorrect_mask(self):
        return (self.movement_metrics.player_task_decision_array*self.raw_data.agent_task_decision_array == -1)
    
    @property
    def score_metric(self, score_type):
        '''
        Returns a dictionary of wins, incorrects, and indecisions
        '''
        score_mask_dict = {'wins': self.win_mask, 'incorrects': self.incorrect_mask, 'indecisions': self.indecision_mask}
        return np.count_nonzero(score_mask_dict[score_type], axis=1)    
    
    @property
    def trial_results(self):
        ans = np.zeros((self.exp_info.num_task_blocks,self.exp_info.num_task_trials))*np.nan
        ans[self.win_mask] = 1
        ans[self.indecision_mask] = 2
        ans[self.incorrect_mask] = 3  
        return ans
    
    @property
    def exp2_points_scored(self):
        points_c0 = self.score_metric['wins'][0]
        points_c1 = self.score_metric['wins'][1] - self.score_metric['incorrects'][1]
        points_c2 = self.score_metric['wins'][2] - self.score_metric['indecisions'][2] 
        points_c3 = self.score_metric['wins'][3] - self.score_metric['incorrects'][3] - self.score_metric['indecisions'][3] 
        
        return np.array([points_c0,points_c1,points_c2,points_c3])
    
    @property
    def both_reached_mask(self):
        ans = np.logical_and(~self.indecision_mask!=0, 
                                           self.raw_data.agent_task_decision_array!=0)
        return ans
    
    def wins_when_both_reach(self,perc=True):
        win_and_both_reached_mask = self.movement_metrics.both_reached_mask*self.win_mask
        if perc:
            # Number of wins when both reached, divided by the number of times both reach 
            return (np.count_nonzero(win_and_both_reached_mask,axis=1)/np.count_nonzero(self.movement_metrics.both_reached_mask))*100
        else:
            return np.count_nonzero(win_and_both_reached_mask,axis=1)
        
    def incorrects_when_both_reach(self,perc=True):
        incorrect_and_both_reached_mask = self.movement_metrics.both_reached_mask*self.incorrect_mask
        if perc:
            # Number of wins when both reached, divided by the number of times both reach 
            return (np.count_nonzero(incorrect_and_both_reached_mask,axis=1)/np.count_nonzero(self.movement_metrics.both_reached_mask))*100
        else:
            return np.count_nonzero(incorrect_and_both_reached_mask,axis=1)
    
    
class ReactionGambleMetrics:
    def __init__(self, exp_info: ExperimentInfo, raw_data: RawData, 
                 movement_metrics: MovementMetrics, score_metrics: ScoreMetrics):
        self.exp_info = exp_info
        self.raw_data = raw_data
        self.movement_metrics = movement_metrics
        self.score_metrics = score_metrics
        self.reaction_time_threshold = 200
        
    def set_reaction_time_threshold(self, threshold, **kwargs):
        reaction_time_subtraction = kwargs.get('reaction_time_subtraction')
        if not isinstance(threshold,int):
            self.reaction_time_threshold = np.nanmean(self.movement_metrics.reaction_times) - reaction_time_subtraction
        else:
            self.reaction_time_threshold = threshold
    
    def react_guess_mask(self, react_or_guess):
        guess_mask =  (
            ((self.movement_metrics.movement_onset_times(task='task')
            - self.raw_data.agent_task_leave_time) 
            <= self.reaction_time_threshold)
            | self.incorrect_mask
        )
        react_mask = ~guess_mask
        
        if react_or_guess == 'react':
            return react_mask
        elif react_or_guess == 'guess':
            return guess_mask
    
    def total_reaction_guess(self, react_or_guess):
        return np.count_nonzero(self.react_guess_mask(react_or_guess),axis=1)
    
    def react_guess_results(self, react_or_guess):
        return mask_array(self.score_metrics.trial_results, self.react_guess_mask(react_or_guess))
    
    def react_guess_score_metric_dict(self, react_or_guess):
        wins = np.count_nonzero(self.react_guess_results(react_or_guess)==1,axis=1)
        indecisions = np.count_nonzero(self.react_guess_results(react_or_guess)==2,axis=1)
        incorrects = np.count_nonzero(self.react_guess_results(react_or_guess)==3,axis=1)
        return {'wins':wins, 'incorrects':incorrects, 'indecisions':indecisions}
    
    def react_guess_incorrects(self, react_or_guess):
        return np.count_nonzero(self.react_guess_results(react_or_guess)==2,axis=1)
    
    def react_guess_indecisions(self, react_or_guess):
        return np.count_nonzero(self.react_guess_results(react_or_guess)==3,axis=1)
    
    def reaction_guess_that_were_score_metric(self, metric, react_or_guess):
        '''
        Out of the x reaction/guess decisions, how many were wins, indecisions, incorrects 
        '''
        numerator = self.react_guess_score_metric_dict(react_or_guess)[metric] # 
        denominator = self.total_reaction_guess(react_or_guess)
        
        return np.divide(numerator,denominator, # gamble_wins/total_gambles
                    out=np.zeros_like(numerator)*np.nan,where=denominator!=0)*100
        
        
    def score_metric_that_were_reaction_guess(self,metric, react_or_guess):
        '''
        Out of the wins, indecisions, incorrects, how many were reactions and guesses
        '''
        numerator = self.react_guess_score_metric_dict(react_or_guess)[metric] # Total reaction/guess wins, indecions, incorrects   
        denominator = self.score_metrics.score_metric[metric] # Total wins, indecisions, incorrects
        
        return np.divide(numerator,denominator, # gamble_wins/total_gambles
                    out=np.zeros_like(numerator)*np.nan,where=denominator!=0)*100
class SubjectBuilder:
    def __init__(self,subject,experiment,num_task_blocks,num_task_trials_initial,num_reaction_blocks,
                 num_reaction_trials,num_timing_trials, select_trials,
                 reaction_xypos_data,reaction_dist_data, reaction_xyvelocity_data, 
                 reaction_speed_data, reaction_trial_type_array, reaction_trial_start, 
                 agent_reaction_decision_array, agent_reaction_leave_time, interval_trial_start, interval_reach_time, coincidence_trial_start, coincidence_reach_time, 
                 task_xypos_data, task_dist_data, task_xyvelocity_data, task_speed_data, agent_task_decision_array, agent_task_leave_time,
                 ):
        self.exp_info = ExperimentInfo(subject=subject, experiment=experiment, num_task_blocks=num_task_blocks, 
                                  num_task_trials_initial=num_task_trials_initial, num_reaction_blocks=num_reaction_blocks,
                                  num_reaction_trials=num_reaction_trials, num_timing_trials=num_timing_trials,
                                  select_trials=select_trials)
        
        self.raw_data = RawData(exp_info = self.exp_info,
                           reaction_xypos_data=reaction_xypos_data,reaction_dist_data=reaction_dist_data, 
                           reaction_xyvelocity_data=reaction_xyvelocity_data, reaction_speed_data=reaction_speed_data, 
                           reaction_trial_type_array=reaction_trial_type_array, reaction_trial_start=reaction_trial_start, 
                           agent_reaction_decision_array=agent_reaction_decision_array, agent_reaction_leave_time=agent_reaction_leave_time, 
                           interval_trial_start=interval_trial_start, interval_reach_time=interval_reach_time, 
                           coincidence_trial_start=coincidence_trial_start, coincidence_reach_time=coincidence_reach_time, 
                           task_xypos_data=task_xypos_data, task_dist_data=task_dist_data, 
                           task_xyvelocity_data=task_xyvelocity_data, task_speed_data=task_speed_data, 
                           agent_task_decision_array=agent_task_decision_array, agent_task_leave_time=agent_task_leave_time,
                           )
        
        self.movement_metrics = MovementMetrics(exp_info=self.exp_info, raw_data=self.raw_data)
        
        
    
        
        
        
    
    
    def create_result_of_trial_array(self):
        '''
        1: Win
        2: Indecision
        3: Incorrect
        '''
        self.trial_results = np.zeros((self.num_task_blocks,self.num_task_trials))*np.nan
        self.trial_results[self.win_mask] = 1
        self.trial_results[self.indecision_mask] = 2
        self.trial_results[self.incorrect_mask] = 3  
        self.trial_results_check = self.mask_array(self.trial_results,self.check_init_reach_direction)
        self.num_trials_corrected = np.count_nonzero(~np.isnan(self.trial_results_check),axis=1)
        
    def remove_reaction_time_nans(self,arr=None,mask=None):
        '''
        This actually removes the values of the array, as opposed to replacing False values with 
        NanN.
        
        Need to do this to plot boxplots I think 
        '''
        
        gamble_nanmask = np.isnan(self.gamble_reaction_time_all)
        self.gamble_reaction_time_only_gamble        = self.gamble_reaction_time_all[2,:][~gamble_nanmask[2,:]]
        self.gamble_reaction_time_mixed              = self.gamble_reaction_time_all[0,:][~gamble_nanmask[0,:]]
        
        react_nanmask = np.isnan(self.react_reaction_time_all)
        self.react_reaction_time_only_react         = self.react_reaction_time_all[1,:][~react_nanmask[1,:]]
        self.react_reaction_time_mixed              = self.react_reaction_time_all[0,:][~react_nanmask[0,:]]
    
    def get_linear_movement_onset_time(self,data):
        nan_mask = np.isnan(data)
        data[nan_mask] = 0
        max_force = np.nanmax(data,axis=2)
        index_max_force = np.nanargmax(data,axis=2)

        max25 = 0.25*max_force
        max75 = 0.75*max_force
        max25_array = np.repeat(max25[...,np.newaxis],data.shape[-1],axis=2) # Extend the array out for all the timesteps (either 8000 for reaction data, or 2000 for task data)
        max75_array = np.repeat(max75[...,np.newaxis],data.shape[-1],axis=2)

        indices = np.arange(0,data.shape[-1],1)
        tile_shape = list(data.shape[:-1])
        tile_shape.append(1)
        indices_tiled = np.tile(indices,tuple(tile_shape)) # Get the first two axes (going to be 3,100 for reaction or 4,80 for task for exp2)
        index_max_force_array = np.repeat(index_max_force[...,np.newaxis],data.shape[-1],axis=2)
        max_force_mask = indices_tiled < index_max_force_array
        max25_timepoints = np.argmin(abs(data*max_force_mask - max25_array),axis=2).astype('float')
        max75_timepoints = np.argmin(abs(data*max_force_mask - max75_array),axis=2).astype('float')

        max25[max25==0] = np.nan
        max75[max75==0] = np.nan
        max25_timepoints[max25_timepoints==0] = np.nan
        max75_timepoints[max75_timepoints==0] = np.nan

        x1vals = max25_timepoints
        x2vals = max75_timepoints
        y1vals = max25
        y2vals = max75
        # if x1vals - x2vals == 0:
        #     return np.nan
        slopes = (y2vals - y1vals)/(x2vals - x1vals)
        intercepts = y2vals - slopes*x2vals
        time_at_zero = -intercepts/slopes
        return time_at_zero

    def analyze_data(self,**kwargs):
        #* decide on sd for reaction time
            
        self.num_stds_for_reaction_time     = kwargs.get('num_stds_for_reaction_time') 
        self.task_leave_time_metric_name    = kwargs.get('task_leave_time_metric_name')
        self.task_movement_time_metric_name = kwargs.get('task_movement_time_metric_name')
        
        #* Set the actual reaction time metric that I'll be using, defaults to using position 
        self.reaction_time_metric_name          = kwargs.get('reaction_time_metric_name')
        self.reaction_movement_time_metric_name = kwargs.get('reaction_movement_time_metric_name')
        
        
        
        #* Find leave,reaction,movement times
        self.find_leave_times()
        
        self.player_task_leave_time         = getattr(self,self.task_leave_time_metric_name).astype(float)
        self.player_task_leave_time_nan     = self.player_task_leave_time
        self.player_task_leave_time_nan[self.player_task_leave_time==self.big_num] = np.nan     
        self.player_task_leave_time_nan[self.player_task_leave_time==0] = np.nan     
        self.player_task_movement_time      = getattr(self,self.task_movement_time_metric_name).astype(float)
        self.player_task_movement_time_nan  = self.player_task_movement_time
        self.player_task_movement_time_nan[self.player_task_movement_time==self.big_num] = np.nan          
        self.reaction_time                  = getattr(self,self.reaction_time_metric_name).astype(float)
        self.reaction_time_cutoff_mask      = (self.reaction_time>600)|(self.reaction_time<170)
        self.reaction_time                  = self.mask_array(self.reaction_time,~self.reaction_time_cutoff_mask)
        self.reaction_movement_time         = getattr(self,self.reaction_movement_time_metric_name).astype(float)
        self.reaction_movement_time[self.reaction_movement_time>450] = np.nan
        self.player_minus_agent_task_leave_time = self.player_task_leave_time - self.agent_task_leave_time
        self.player_minus_agent_task_leave_time_nan = self.player_task_leave_time_nan - self.agent_task_leave_time
        
        self.find_correct_initial_decisions()
        
        
        self.win_mask          = np.logical_or(self.player_task_decision_array*self.agent_task_decision_array == 1, 
                                               np.logical_and(self.player_task_decision_array!=0, self.agent_task_decision_array==0))
        self.indecision_mask   = (self.player_task_decision_array == 0)
        self.incorrect_mask    = (self.player_task_decision_array*self.agent_task_decision_array == -1)
        self.both_decided_mask = (abs(self.player_task_decision_array*self.agent_task_decision_array) == 1)
        
        self.win_mask_corrected                   = self.mask_array(self.win_mask.astype(float),self.check_init_reach_direction)
        self.indecision_mask_corrected            = self.mask_array(self.indecision_mask.astype(float),self.check_init_reach_direction)
        self.incorrect_mask_corrected             = self.mask_array(self.incorrect_mask.astype(float),self.check_init_reach_direction)
        self.both_decided_mask_corrected          = self.mask_array(self.both_decided_mask.astype(float),self.check_init_reach_direction)
        
        self.create_result_of_trial_array()
        
        
        #* Only parse reaction time data this way if it's experiment 2, because I switched the reaction time task 
        if self.experiment == 'Exp2':
            # Parse Reaction Time Data
            self.parse_reaction_task_trials_exp2()
            self.calculate_reaction_repeats_alternates_exp2()
            # self.remove_reaction_time_nans()
            self.adjusted_player_reaction_time    = np.nanmean(self.react_reaction_time_only_react) - \
                                                               self.num_stds_for_reaction_time*np.nanstd(self.react_reaction_time_only_react) # CONSERVATIVE REACTION TIME
            
        elif self.experiment == 'Exp1':  
            self.adjusted_player_reaction_time = np.nanmean(self.reaction_time) - self.num_stds_for_reaction_time*np.nanstd(self.reaction_time)
                                                      
        #------------------------------------------------------------------------------------------------------------------
        # Task Indecision, Wins, Incorrects
        self.calc_wins_indecisions_incorrects()
        
        # Decision and Reach  Times on Inleaves
        self.leave_and_reach_times_on_wins_incorrects_indecisions()
                
        # Gamble and Reaction Calculations
        # self.reaction_gamble_calculations()
        self.estimate_reaction_latency()
        self.reaction_gamble_calculations()
        
        # Wins when both decide
        self.wins_when_both_decide()
        
        # Binned metrics
        # self.binned_metrics()
        
        # Estimate true reaction time during task 
  
    
    def find_leave_times(self):
        self.big_num = 100000
        def _get_target_reach_times(xydata):
            # Argmax finds the first id where the condition is true
            ans1 = np.argmax(np.sqrt((xydata[...,0]-self.target1x)**2 + 
                                     (xydata[...,1]-self.target1y)**2) < self.target1_radius,axis=2) # Find when people enter the right target
            ans2 = np.argmax(np.sqrt((xydata[...,0]-self.target2x)**2 + 
                                     (xydata[...,1]-self.target2y)**2) < self.target2_radius,axis=2)
            # DOING THIS WAY instaed of doing np.maximum, bc sometimes people end up reahcing BOTH targets, so np.argmax returns a non-zero value (NOT RELEVANT FOR REACTION)
            # Therefore, i can't take the maximum in that case, so I need to make 0 equal 100000 so I can then take the minimum
            ans1[ans1 == 0] = self.big_num
            ans2[ans2 == 0] = self.big_num
            return ans1,ans2
        #*----------------------- REACTION --------------------------
        #* Get leave time (pos, velocity lin, velocity thresh)
        self.player_pos_reaction_leave_time                 = np.argmax(self.reaction_dist_data > self.start_radius,axis=2)
        self.player_velocity_reaction_leave_time_thresh     = np.argmax(self.reaction_speed_data > 0.05,axis=2)
        self.player_velocity_reaction_leave_time_linear     = self.get_linear_movement_onset_time(self.reaction_speed_data)
        
        #* In exp1, the reaction stimulus is at the trial start time, so don't subtract off anything
        if self.experiment == 'Exp1':
            #* Get reach time ids
            self.reaction_enter_right_target_id,\
            self.reaction_enter_left_target_id = _get_target_reach_times(self.reaction_xypos_data)
            self.reaction_enter_right_target_id = self.reaction_enter_right_target_id[2]
            self.reaction_enter_left_target_id = self.reaction_enter_left_target_id[2]
            self.player_reaction_reach_time                     = np.minimum(self.reaction_enter_right_target_id,
                                                                             self.reaction_enter_left_target_id).astype(float)
            self.player_reaction_reach_time[self.player_reaction_reach_time==self.big_num] = np.nan
            self.player_pos_reaction_time                       = self.player_pos_reaction_leave_time[2] #! Last row is the actual reaction, first two are timing for exp1
            self.player_pos_reaction_movement_time              = self.player_reaction_reach_time - self.player_pos_reaction_leave_time[2]
            self.player_velocity_reaction_time_thresh           = self.player_velocity_reaction_leave_time_thresh[2] 
            self.player_velocity_reaction_movement_time_thresh  = self.player_reaction_reach_time - self.player_velocity_reaction_leave_time_thresh[2]
            self.player_velocity_reaction_movement_time_thresh[self.player_velocity_reaction_movement_time_thresh>450]  = np.nan
            self.player_velocity_reaction_time_linear           = self.player_velocity_reaction_leave_time_linear[2]  
            self.player_velocity_reaction_movement_time_linear  = self.player_reaction_reach_time - self.player_velocity_reaction_leave_time_linear[2] 
            
        #* In exp2, the reaction stimulus is the agent
        elif self.experiment == 'Exp2':
            #* Get reach time ids
            self.reaction_enter_right_target_id,\
            self.reaction_enter_left_target_id = _get_target_reach_times(self.reaction_xypos_data)
            self.player_reaction_reach_time                     = np.minimum(self.reaction_enter_right_target_id,
                                                                             self.reaction_enter_left_target_id).astype(float)
            self.player_pos_reaction_time                       = self.player_pos_reaction_leave_time - self.agent_reaction_leave_time
            self.player_pos_reaction_movement_time              = self.player_reaction_reach_time - self.player_pos_reaction_leave_time
            self.player_velocity_reaction_time_thresh           = self.player_velocity_reaction_leave_time_thresh - self.agent_reaction_leave_time
            self.player_velocity_reaction_movement_time_thresh  = self.player_reaction_reach_time - self.player_velocity_reaction_leave_time_thresh
            self.player_velocity_reaction_time_linear           = self.player_velocity_reaction_leave_time_linear - self.agent_reaction_leave_time
            self.player_velocity_reaction_movement_time_linear  = self.player_reaction_reach_time - self.player_velocity_reaction_leave_time_linear 
        
        self.player_reaction_decision_array = np.zeros((self.reaction_enter_right_target_id.shape))*np.nan
        #* Determine the decision array based on target selection or indecision
        self.player_reaction_decision_array[self.reaction_enter_right_target_id<self.reaction_enter_left_target_id] = 1 # Player selected right target
        self.player_reaction_decision_array[self.reaction_enter_left_target_id<self.reaction_enter_right_target_id] = -1
        self.player_reaction_decision_array[self.player_reaction_reach_time>1500] = 0
        self.player_reaction_decision_array[self.player_reaction_reach_time==0] = 0
        
        #* -------------------------------------------TASK ------------------------------------------------------------
        # Same for all metrics
        #* Get reach time ids
        self.task_enter_right_target_id,\
        self.task_enter_left_target_id = _get_target_reach_times(self.task_xypos_data)
        self.player_task_reach_time  = np.minimum(self.task_enter_right_target_id,self.task_enter_left_target_id).astype(float)
        #* Determine the decision array based on target selection or indecision
        self.player_task_decision_array = np.zeros((self.num_task_blocks,self.num_task_trials))*np.nan
        self.player_task_decision_array[self.task_enter_right_target_id<self.task_enter_left_target_id] = 1 # Player selected right target
        self.player_task_decision_array[self.task_enter_left_target_id<self.task_enter_right_target_id] = -1
        self.player_task_decision_array[self.player_task_reach_time>1500] = 0
        self.player_task_decision_array[self.player_task_reach_time==self.big_num] = 0
        
        #* Make this nan AFTER getting decision array, so it counts those as indecisions
        self.player_task_reach_time[self.player_task_reach_time==self.big_num] = np.nan # If task reach time is 0, then they never reached a target, so can't calculate movement time from that trial, so make it nan
        
        #* Using position
        self.player_pos_task_leave_time                 = np.argmax(self.task_dist_data > self.start_radius,axis=2)
        self.player_minus_agent_pos_task_leave_time     = self.player_pos_task_leave_time - self.agent_task_leave_time
        self.player_pos_task_movement_time              = self.player_task_reach_time - self.player_pos_task_leave_time
        #* Using velocity with 0.05 threshold
        self.player_velocity_task_leave_time_thresh        = np.argmax(self.task_speed_data > 0.05,axis=2)
        self.player_minus_agent_velocity_task_leave_time   = self.player_velocity_task_leave_time_thresh - self.agent_task_leave_time
        self.player_velocity_task_movement_time_thresh     = self.player_task_reach_time - self.player_velocity_task_leave_time_thresh
        #* Using velocity with linear estimation
        self.player_velocity_task_leave_time_linear        = self.get_linear_movement_onset_time(self.task_speed_data)
        self.player_velocity_task_time_linear              = self.player_velocity_task_leave_time_linear - self.agent_task_leave_time
        self.player_velocity_task_movement_time_linear     = self.player_task_reach_time - self.player_velocity_task_leave_time_linear 
    
    def parse_reaction_task_trials_exp2(self):
        
        #* Split into react and gamble trials for first block
        self.reaction_gamble_mask                 = self.reaction_trial_type_array == 0
        self.reaction_react_mask                  = self.reaction_trial_type_array == 1
        
        # * Reaction Time        
        self.react_reaction_time_all          = self.reaction_time[self.reaction_react_mask].reshape(2,50)   
        self.gamble_reaction_time_all         = self.reaction_time[self.reaction_gamble_mask].reshape(2,50) 
        self.react_reaction_time_mixed        = self.react_reaction_time_all[0,:]
        self.react_reaction_time_only_react   = self.react_reaction_time_all[1,:]
        self.gamble_reaction_time_mixed       = self.gamble_reaction_time_all[0,:]
        self.gamble_reaction_time_only_gamble = self.gamble_reaction_time_all[1,:]
        
        # * Movement time
        self.react_movement_time_all          = self.reaction_movement_time[self.reaction_react_mask].reshape(2,50)   
        self.gamble_movement_time_all         = self.reaction_movement_time[self.reaction_gamble_mask].reshape(2,50) 
        self.react_movement_time_mixed        = self.react_movement_time_all[0,:]
        self.react_movement_time_only_react   = self.react_movement_time_all[1,:]
        self.gamble_movement_time_mixed       = self.gamble_movement_time_all[0,:]
        self.gamble_movement_time_only_gamble = self.gamble_movement_time_all[1,:]
        
        # assert (self.reaction_time_all == self.mask_array(self.reaction_time,self.reaction_react_mask))
    def calculate_reaction_repeats_alternates_exp2(self):
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
        self.reaction_time_mixed = self.reaction_time[0,:]
        self.gamble_reaction_time_repeat = self.mask_array(self.reaction_time_mixed,self.gamble_repeat_mask)
        self.gamble_reaction_time_switch = self.mask_array(self.reaction_time_mixed,self.gamble_switch_mask)
        # Get the reaction times of reactions on repeats and mixeds
        self.react_reaction_time_repeat  = self.mask_array(self.reaction_time_mixed,self.react_repeat_mask)
        self.react_reaction_time_switch  = self.mask_array(self.reaction_time_mixed,self.react_switch_mask)      
        
    def calc_wins_indecisions_incorrects(self):
        self.player_wins             = np.count_nonzero(self.win_mask,axis=1)
        self.player_indecisions      = np.count_nonzero(self.indecision_mask,axis=1)
        self.player_incorrects       = np.count_nonzero(self.incorrect_mask,axis=1)
        
        self.player_perc_wins        = (self.player_wins/self.num_task_trials)*100
        self.player_perc_indecisions = (self.player_indecisions/self.num_task_trials)*100
        self.player_perc_incorrects  = (self.player_incorrects/self.num_task_trials)*100
        
        self.player_wins_corrected             = np.nansum(self.win_mask_corrected,axis=1)
        self.player_indecisions_corrected      = np.nansum(self.indecision_mask_corrected,axis=1)
        self.player_incorrects_corrected       = np.nansum(self.incorrect_mask_corrected,axis=1)
        
        self.player_perc_wins_corrected        = (self.player_wins_corrected/self.num_trials_corrected)*100
        self.player_perc_indecisions_corrected = (self.player_indecisions_corrected/self.num_trials_corrected)*100
        self.player_perc_incorrects_corrected  = (self.player_incorrects_corrected/self.num_trials_corrected)*100
        
        # self.player_perc_wins_corrected = (self.player_wins/self.num_task_trials)*100
        
        if self.experiment == 'Exp2':
           self.points_c0 = self.player_wins[0]
           self.points_c1 = self.player_wins[1] - self.player_incorrects[1]
           self.points_c2 = self.player_wins[2] - self.player_indecisions[2] 
           self.points_c3 = self.player_wins[3] - self.player_incorrects[3] - self.player_indecisions[3] 
           self.player_points_scored = np.array([self.points_c0,self.points_c1,self.points_c2,self.points_c3])
        
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
        
    def find_correct_initial_decisions(self):
        # * Find initial reach diretion using x positino
        self.time_for_init_reach_direction = self.player_task_leave_time_nan + 30 # Look 30 milliseconds later
        indices = np.arange(0,self.task_xypos_data[...,0].shape[-1],1,dtype=float) # Use arange to create an array of INDICES, will mask this later to get the exact point we want
        tile_shape = list(self.task_xypos_data[...,0].shape[:-1]) # get the shape of the desired array but not the last or second to last (bc last is 0 or 1 for x and y)
        tile_shape.append(1) # Append 1 to the list of the tile_shape
        self.indices_tiled = np.tile(indices,tuple(tile_shape))  # Turn tile shape back to tuple so it works with np tile, then tile the indices we got in same shape of data
        
        # Make the times we want have the same shape as data
        self.index_init_reach_direction_array = np.repeat(self.time_for_init_reach_direction[...,np.newaxis],
                                                     self.task_xypos_data[...,0].shape[-1],axis=2)  
        # Check where the indices are equal to the reach direction index
        self.init_reach_pos_mask = self.indices_tiled == self.index_init_reach_direction_array.astype(int) # 
        # Use the mask (that puts true at the index where the indices are equal to the reach direction time)
        self.init_reach_posx  = self.mask_array(self.task_xypos_data[...,0],self.init_reach_pos_mask)
        self.init_reach_posy  = self.mask_array(self.task_xypos_data[...,1],self.init_reach_pos_mask) 
        # Get mask for when their initial direction went to the right
        self.init_reach_posx_single_timepoint = np.nanmax(self.init_reach_posx*self.init_reach_pos_mask,axis=2)
        self.right_mask = self.init_reach_posx_single_timepoint - self.startx > 0
        
        # self.right_mask = self.init_reach_posx[self.init_reach_pos_mask].reshape(self.init_reach_pos_mask.shape[:-1]) - self.startx > 0 # DID THEY REACH RIGHTWARDS OF THE START TARGET INITIALLY
        # Get the initial reach direction (1 for right, -1 for left)
        self.init_reach_direction = np.zeros((self.num_task_blocks,self.num_task_trials))
        self.init_reach_direction[self.right_mask] = 1
        self.init_reach_direction[~self.right_mask] = -1
        # Get task decision array without indecisions, just to check
        self.player_task_decision_array_check = np.zeros((self.player_task_decision_array.shape))*np.nan
        self.player_task_decision_array_check[self.task_enter_right_target_id<self.task_enter_left_target_id] = 1 # Player selected right target
        self.player_task_decision_array_check[self.task_enter_left_target_id<self.task_enter_right_target_id] = -1
        # Compare that with the target they selected to check
        self.check_init_reach_direction = self.init_reach_direction == self.player_task_decision_array_check
        
        # * Determine if initial reach angle is the same as the agent's target selection and classify incorrects and corrects
        self.correct_init_decision_mask = self.init_reach_direction == self.agent_task_decision_array   
    
    def estimate_reaction_latency(self):
        self.find_correct_initial_decisions()
        
        # * Find player-agent times for corrects and incorrects and take mean
        # Get the number of correct and incorrect initial movement decisions
        
        # If i want to take their mean across all the blocks do this
        self.player_minus_agent_on_errors_all = self.player_minus_agent_task_leave_time[~self.correct_init_decision_mask]
        self.mean_player_minus_agent_on_errors_all = np.nanmean(self.player_minus_agent_on_errors_all)
        self.player_minus_agent_on_corrects_all = self.player_minus_agent_task_leave_time[self.correct_init_decision_mask]
        self.mean_player_minus_agent_on_corrects_all = np.nanmean(self.player_minus_agent_on_corrects_all)
        
        # Estimate mu_{s}
        self.phat_correct_all= np.count_nonzero(self.correct_init_decision_mask)/(self.num_task_trials*self.num_task_blocks)
        self.phat_error_all   = np.count_nonzero(~self.correct_init_decision_mask)/(self.num_task_trials*self.num_task_blocks)
        self.mhat_correct_all = self.mean_player_minus_agent_on_corrects_all
        self.mhat_error_all = self.mean_player_minus_agent_on_errors_all
        self.mu_s_all = (self.phat_correct_all*self.mhat_correct_all - self.phat_error_all*self.mhat_error_all)/(self.phat_correct_all - self.phat_error_all)
        
        self.player_leave_time_on_errors_all = self.player_task_leave_time[~self.correct_init_decision_mask]
        self.mean_player_leave_time_on_errors_all = np.nanmean(self.player_leave_time_on_errors_all)
        self.player_leave_time_on_corrects_all = self.player_task_leave_time[self.correct_init_decision_mask]
        self.mean_player_leave_time_on_corrects_all = np.nanmean(self.player_leave_time_on_corrects_all)
        
        # Estimate mu_{s}
        self.phat_correct_all_leave = np.count_nonzero(self.correct_init_decision_mask)/(self.num_task_trials*self.num_task_blocks)
        self.phat_error_all_leave   = np.count_nonzero(~self.correct_init_decision_mask)/(self.num_task_trials*self.num_task_blocks)
        self.mhat_correct_all_leave = self.mean_player_leave_time_on_corrects_all
        self.mhat_error_all_leave   = self.mean_player_leave_time_on_errors_all
        self.mu_s_all_leave = (self.phat_correct_all_leave*self.mhat_correct_all_leave - self.phat_error_all_leave*self.mhat_error_all_leave)/(self.phat_correct_all_leave - self.phat_error_all)
        
        self.phat_correct = np.count_nonzero(self.correct_init_decision_mask,axis=1)/self.num_task_trials
        self.phat_error   = np.count_nonzero(~self.correct_init_decision_mask,axis=1)/self.num_task_trials
        # Leave time for each block
        self.player_leave_time_on_errors = self.mask_array(self.player_task_leave_time,~self.correct_init_decision_mask)
        self.mean_player_leave_time_on_errors = np.nanmean(self.player_leave_time_on_errors,axis=1)
        self.player_leave_time_on_corrects = self.mask_array(self.player_task_leave_time,self.correct_init_decision_mask)
        self.mean_player_leave_time_on_corrects = np.nanmean(self.player_leave_time_on_corrects,axis=1)
        # Pplayer minus agent leave time for each block 
        self.player_minus_agent_on_errors = self.mask_array(self.player_minus_agent_task_leave_time,~self.correct_init_decision_mask)
        self.mean_player_minus_agent_on_errors = np.nanmean(self.player_minus_agent_on_errors,axis=1)
        self.player_minus_agent_on_corrects = self.mask_array(self.player_minus_agent_task_leave_time,self.correct_init_decision_mask)
        self.mean_player_minus_agent_on_corrects = np.nanmean(self.player_minus_agent_on_corrects,axis=1)
        
        # Estimate mu_{s}
        self.mhat_correct = self.mean_player_minus_agent_on_corrects
        self.mhat_error = self.mean_player_minus_agent_on_errors
        self.mu_s = (self.phat_correct*self.mhat_correct - self.phat_error*self.mhat_error)/(self.phat_correct - self.phat_error)
        
        #GEt mu_s for leave time instead of player minus agent
        self.player_leave_time_on_errors = self.mask_array(self.player_task_leave_time,~self.correct_init_decision_mask)
        self.mean_player_leave_time_on_errors = np.nanmean(self.player_leave_time_on_errors,axis=1)
        self.player_leave_time_on_corrects = self.mask_array(self.player_task_leave_time,self.correct_init_decision_mask)
        self.mean_player_leave_time_on_corrects = np.nanmean(self.player_leave_time_on_corrects,axis=1)
        
        # Estimate mu_{s}
        self.mhat_correct_alternate = self.mean_player_leave_time_on_corrects
        self.mhat_error_alternate = self.mean_player_leave_time_on_errors
        self.mu_s_alternate = (self.phat_correct*self.mhat_correct_alternate - self.phat_error*self.mhat_error_alternate)/(self.phat_correct - self.phat_error)
    
    def select_true_reaction_time(self):
        # Pick an initial reaction cutoff
        if self.experiment == 'Exp1':
            self.reaction_cutoff = np.nanmean(self.reaction_time) + 20
        elif self.experiment == 'Exp2':
            self.reaction_cutoff = np.nanmean(self.react_reaction_time_only_react) + 20

        #* See if number of gamble initial decision direction is close to 50%
        flag = True
        while flag:
            # Get the gamble mask
            self.gamble_mask = self.player_minus_agent_task_leave_time < self.reaction_cutoff
            # Apply gamble mask on correct_init_decision mask to get the correct decision mask for only gamble decisions
            self.gamble_init_decision = self.correct_init_decision_mask[self.gamble_mask]
            # Out of all the gambles, how many were correct (True)
            self.phat_gambles = np.count_nonzero(self.gamble_init_decision)/len(self.gamble_init_decision)
            if self.phat_gambles>0.55:
                self.reaction_cutoff = self.reaction_cutoff - 1
            else:
                flag=False
        return self.reaction_cutoff
        
    def reaction_gamble_calculations(self):
     
        if self.reaction_time_metric_name == 'player_pos_reaction_time':
            self.reaction_time_threshold = 230
        else:
            self.reaction_time_threshold = 200
        # Gamble calculations
        if True:
            # Create mask and get the total number of gamble decisions
            self.gamble_task_mask = ((self.player_task_leave_time-self.agent_task_leave_time) <= self.reaction_time_threshold)|self.incorrect_mask
            self.total_gambles = np.count_nonzero(self.gamble_task_mask,axis=1)
            
            # Get the leave time and reach target time
            self.player_gamble_task_leave_time   = self.mask_array(self.player_task_leave_time,self.gamble_task_mask)
            self.player_gamble_task_reach_time   = self.mask_array(self.player_task_reach_time,self.gamble_task_mask)
            self.agent_gamble_task_leave_time    = self.mask_array(self.agent_task_leave_time,self.gamble_task_mask)
            self.agent_gamble_task_reach_time    = self.mask_array(self.agent_task_reach_time,self.gamble_task_mask)
            self.player_minus_agent_gamble_task_leave_time = self.player_gamble_task_leave_time - self.agent_gamble_task_leave_time

            # Wins, indecisions,incorrects
            self.gamble_results_array = self.mask_array(self.trial_results,self.gamble_task_mask)
            self.gamble_wins          = np.count_nonzero(self.gamble_results_array==1,axis=1)
            self.gamble_indecisions   = np.count_nonzero(self.gamble_results_array==2,axis=1)
            self.gamble_incorrects    = np.count_nonzero(self.gamble_results_array==3,axis=1)
            
            # Get percent gamble decisions
            self.perc_gamble_decisions = self.total_gambles/self.num_task_trials*100
            
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
            self.reaction_task_mask   = ~self.gamble_task_mask
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
            self.perc_reaction_decisions = self.total_reactions/self.num_task_trials*100
            
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
        self.player_perc_both_reached_wins = np.zeros((self.num_task_blocks))
        self.agent_both_reached_wins = np.zeros((self.num_task_blocks))
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
        combine_reaction_and_gamble_decisions = np.hstack((self.agent_reaction_task_leave_time, (self.player_gamble_task_leave_time - gamble_delay))) 
        self.player_decision_times = np.nanmedian(combine_reaction_and_gamble_decisions,axis=1)
        
    def predict_stopping_times(self,gamble_delay,weird_delay=0):    
        self._predict_stopping_time_from_reactions_gambles(weird_delay=weird_delay)
    
    def _predict_stopping_time_from_reactions_gambles(self,weird_delay):
        '''
        Using the percentage reactions and gambles to predict the stopping time
        '''
        timesteps = np.arange(1000,2000,1)
        self.player_stopping_times             = np.zeros((self.num_task_blocks))
        self.player_stopping_times_index       = np.zeros((self.num_task_blocks))
        temp_predicted_perc_reaction_decisions = np.zeros((self.num_task_blocks,len(timesteps)))
        temp_predicted_perc_gamble_decisions   = np.zeros((self.num_task_blocks,len(timesteps)))
        self.predicted_perc_reaction_decisions = np.zeros((self.num_task_blocks))
        self.predicted_perc_gamble_decisions   = np.zeros((self.num_task_blocks))
        react_loss  = np.zeros((self.num_task_blocks,len(timesteps)))
        gamble_loss = np.zeros((self.num_task_blocks,len(timesteps)))
        for j in range(self.num_task_blocks):
            for k,t in enumerate(timesteps):
                temp_predicted_perc_reaction_decisions[j,k] = np.count_nonzero(self.agent_task_leave_time[j,:]<=t+weird_delay)/self.num_task_trials*100
                react_loss[j,k]                             = abs(self.perc_reaction_decisions[j] - temp_predicted_perc_reaction_decisions[j,k])
                temp_predicted_perc_gamble_decisions[j,k]   = np.count_nonzero(self.agent_task_leave_time[j,:]>t+weird_delay)/self.num_task_trials*100
                gamble_loss[j,k]                            = abs(self.perc_gamble_decisions[j] - temp_predicted_perc_gamble_decisions[j,k])
                
            self.player_stopping_times_index[j]       = np.argmin(react_loss[j,:]+gamble_loss[j,:])
            self.predicted_perc_reaction_decisions[j] = temp_predicted_perc_reaction_decisions[j,int(self.player_stopping_times_index[j])]
            self.predicted_perc_gamble_decisions[j]   = temp_predicted_perc_gamble_decisions[j,int(self.player_stopping_times_index[j])]
        
        self.player_stopping_times = self.player_stopping_times_index + np.min(timesteps)
    
    # def predict_stopping_time_from_movement_onset(self,gamble_delay,weird_delay):
    #     stopping_times = np.arange(500,2000,1)
    #     player_stopping_times = np.zeros((self.num_task_blocks))
              
             
    def binned_metrics(self,bin_start = 800,bin_end = 1400, bin_size = 50,cut_off_threshold = 30):
        self.bins = np.arange(bin_start,bin_end,bin_size)
        self.bin_length_each_condition                    = np.zeros((len(self.bins)-1,self.num_task_blocks))
        self.binned_player_task_leave_times               = np.zeros((len(self.bins)-1,self.num_task_blocks,self.num_task_trials))*np.nan 
        self.binned_player_task_decision_array            = np.zeros((len(self.bins)-1,self.num_task_blocks,self.num_task_trials))*np.nan 
        self.binned_agent_task_leave_times                = np.zeros((len(self.bins)-1,self.num_task_blocks,self.num_task_trials))*np.nan 
        self.binned_player_minus_agent_task_leave_time    = np.zeros((len(self.bins)-1,self.num_task_blocks,self.num_task_trials))*np.nan 
        self.binned_agent_task_decision_array             = np.zeros((len(self.bins)-1,self.num_task_blocks,self.num_task_trials))*np.nan 
        self.binned_player_wins                           = np.zeros((len(self.bins)-1,self.num_task_blocks))
        self.binned_player_indecisions                    = np.zeros((len(self.bins)-1,self.num_task_blocks))
        self.binned_player_incorrects                     = np.zeros((len(self.bins)-1,self.num_task_blocks))
        self.mean_binned_player_wins                      = np.zeros((len(self.bins)-1,self.num_task_blocks))
        self.mean_binned_player_indecisions               = np.zeros((len(self.bins)-1,self.num_task_blocks))
        self.mean_binned_player_incorrects                = np.zeros((len(self.bins)-1,self.num_task_blocks))
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
        
        self.bin_cutoff_threshold = kwargs.get('bin_cutoff_threshold',30)
        self.select_trials                  = kwargs.get('select_trials') 
        self.num_stds_for_reaction_time     = kwargs.get('num_stds_for_reaction_time') 
        self.task_leave_time_metric_name    = kwargs.get('task_leave_time_metric_name')
        self.task_movement_time_metric_name = kwargs.get('task_movement_time_metric_name')
        
        #* Set the actual reaction time metric that I'll be using, defaults to using position 
        self.reaction_time_metric_name          = kwargs.get('reaction_time_metric_name')
        self.reaction_movement_time_metric_name = kwargs.get('reaction_movement_time_metric_name')
        
    def analyze_data(self):
        #* Analyze data function for each object 
        for i,o in enumerate(self.objects):
            o.analyze_data(select_trials = self.select_trials, num_stds_for_reaction_time = self.num_stds_for_reaction_time, 
                            task_leave_time_metric_name = self.task_leave_time_metric_name,task_movement_time_metric_name = self.task_movement_time_metric_name,
                            reaction_time_metric_name = self.reaction_time_metric_name, reaction_movement_time_metric_name = self.reaction_movement_time_metric_name)
            self.objects[i] = o
            
        #* Flatten the trial by trial data so I can make a histogram including everyones data
        self.num_task_blocks = self.objects[0].num_task_blocks
        self.num_task_trials = self.objects[0].num_task_trials_initial
        self.flatten_across_all_subjects()
        #* Loop through all attributes, and set the group attribute with all subjects combined
        np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
        for a in dir(self.objects[0]):
            if not a.startswith('__') and not callable(getattr(self.objects[0], a)):
                # with warnings.catch_warnings(record=True) as w:
                #     # Cause all warnings to always be triggered.
                #     warnings.simplefilter("always")
                #     # Trigger a warning.
                #     warnings.warn("deprecated", DeprecationWarning)
                #     # Verify some things
                #     assert len(w) == 1
                #     assert issubclass(w[-1].category, DeprecationWarning)
                #     assert "deprecated" in str(w[-1].message)    
                    
                if isinstance(getattr(self.objects[0],a),np.ndarray):
                    if getattr(self.objects[0],a).shape == getattr(self.objects[1],a).shape:
                        arr = np.array([getattr(o,a) for o in self.objects])
                        setattr(self,a,arr)
                else:
                    arr = np.array([getattr(o,a) for o in self.objects])
                    setattr(self,a,arr)
                    
        #* Assert that there are no double subjects
        for o in self.objects[1:]:
            assert ~(self.objects[0].player_task_leave_time==o.player_task_leave_time).all()
            
    def predict_decision_times(self,gamble_delay):
        for o in self.objects:
            o.predict_decision_times(gamble_delay)
            
    def predict_stopping_times(self,gamble_delay):
        for o in self.objects:
            o.predict_stopping_times(gamble_delay)
    
    def flatten_across_all_subjects(self):
        self.all_player_task_leave_times_each_condition            = self.concatenate_across_subjects('player_task_leave_time')
        self.all_player_task_gamble_leave_times_each_condition     = self.concatenate_across_subjects('player_gamble_task_leave_time')
        self.all_player_task_reaction_leave_times_each_condition   = self.concatenate_across_subjects('player_reaction_task_leave_time')
    
        self.all_agent_task_leave_times_each_condition             = self.concatenate_across_subjects('agent_task_leave_time')
        self.all_agent_task_gamble_leave_times_each_condition      = self.concatenate_across_subjects('agent_gamble_task_leave_time')
        self.all_agent_task_reaction_leave_times_each_condition    = self.concatenate_across_subjects('agent_reaction_task_leave_time')
        
    
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
        
    def concatenate_across_subjects(self,metric):
        '''
        Flattens out the subject dimension to get array of all the subjects 
        
        Usually used for group distributions
        '''
        arr = self.combine_all_subjects(metric)
        temp = np.swapaxes(arr,0,1)
        ans = np.reshape(temp,(self.num_task_blocks,-1))
        return ans
            
    def find_subject(self,metric,comparison_num,comparison_direction):
        '''
        Used to find the subject who's specific value is greater or less than the inputted comparison metric
        '''
        metrics = self.combine_all_subjects(metric)
        if comparison_direction == 'greater than':
            mask = metrics>comparison_num
        if comparison_direction == 'less than':
            mask = metrics<comparison_num
        for i,e in enumerate(mask):
            if e.any():
                print(f'Sub{i}')
    
    def filter_gamble_reaction_under_10(self,metric):
        for o in self.objects:
            for i in range(self.num_task_blocks):
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
        