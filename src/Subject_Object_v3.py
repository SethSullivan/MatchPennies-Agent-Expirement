import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import warnings
from functools import cached_property

SCORE_METRIC_NAMES = ('wins','incorrects','indecisions')

def mask_array(arr,mask):
    '''
    Applies the mask to the array then replaces the 0s with nans
    '''
    new_arr = arr*mask # Apply mask
    new_arr[~mask] = np.nan # Replace the 0s from the mask with np nan
    return new_arr

def collapse_across_subjects(metric,num_blocks):
    '''
    Flattens out the subject dimension to get array of all the subjects 
    '''
    temp = np.swapaxes(metric,0,1)
    ans = np.reshape(temp,(num_blocks,-1))
    return ans

def perc(metric,num_trials=80):
    return (metric/num_trials)*100

class ExperimentInfo:
    def __init__(self, subjects, experiment, num_task_blocks,
                 num_task_trials_initial, num_reaction_blocks,
                 num_reaction_trials, num_timing_trials,
                 select_trials,
                 **kwargs):
        self.subjects = subjects
        self.num_subjects = len(self.subjects)
        #* Target Info 
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
        
        #* Get Experimental Design Data
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
        self.coincidence_reach_time = self.raw_data.coincidence_reach_time
        self.interval_reach_time = self.raw_data.interval_reach_time
        
        self.vel_threshold = kwargs.get('velocity_threshold',0.05)
        self.metric_type   = kwargs.get('metric_type','velocity')

        if self.metric_type not in ['position','velocity','velocity linear']:
            raise ValueError('type should be \'position\', \'velocity\', or \'velocity linear\'')
        
        if self.exp_info.experiment == 'Exp2':
            self.reaction_gamble_mask = self.raw_data.reaction_trial_type_array == 0
            self.reaction_react_mask = self.raw_data.reaction_trial_type_array == 1
        
        self.big_num = 100000
        self.task_enter_right_target_id,self.task_enter_left_target_id = self.right_left_target_ids('task')
    
    def right_left_target_ids(self,task):
        if task == 'reaction':
            xydata = self.raw_data.reaction_xypos_data
        elif task == 'task':
            xydata = self.raw_data.task_xypos_data
        else:
            raise ValueError('task should be \'task\' or \'reaction\'')
        # Argmax finds the first id where the condition is true
        enter_right_target_id = np.argmax(np.sqrt((xydata[...,0]-self.exp_info.target1x)**2 + 
                                    (xydata[...,1]-self.exp_info.target1y)**2) < self.exp_info.target1_radius,axis=3) # Find when people enter the right target
        enter_left_target_id = np.argmax(np.sqrt((xydata[...,0]-self.exp_info.target2x)**2 + 
                                    (xydata[...,1]-self.exp_info.target2y)**2) < self.exp_info.target2_radius,axis=3)
        # DOING THIS WAY instaed of doing np.maximum, bc sometimes people end up reahcing BOTH targets, so np.argmax returns a non-zero value (NOT RELEVANT FOR REACTION)
        # Therefore, i can't take the maximum in that case, so I need to make 0 equal 100000 so I can then take the minimum
        enter_right_target_id[enter_right_target_id == 0] = self.big_num
        enter_left_target_id[enter_left_target_id == 0] = self.big_num
        
        return enter_right_target_id, enter_left_target_id
            
    def target_reach_times(self,task):
        '''
        Calculate when the participants hand position enters the right or left target
        '''
        if task == 'reaction':
            enter_right_target_id, enter_left_target_id = self.right_left_target_ids(task=task)
            reach_times = np.minimum(enter_right_target_id, enter_left_target_id).astype(float)        
            reach_times[reach_times==self.big_num] = np.nan
        
        
            return reach_times
        elif task == 'task':
            reach_times = np.minimum(self.task_enter_right_target_id, self.task_enter_left_target_id).astype(float)        
            reach_times[reach_times==self.big_num] = np.nan
            return reach_times
    
    @cached_property
    def task_decision_array(self):
        #* Determine the decision array based on target selection or indecision
        reach_times = np.minimum(self.task_enter_right_target_id, self.task_enter_left_target_id).astype(float)        
        ans = np.zeros((self.exp_info.num_subjects, self.exp_info.num_task_blocks, self.exp_info.num_task_trials))*np.nan
        
        ans[self.task_enter_right_target_id < self.task_enter_left_target_id] = 1 # Player selected right target
        ans[self.task_enter_left_target_id < self.task_enter_right_target_id] = -1 # Player selected left target
        ans[reach_times > 1500] = 0 # Player failed to select a target
        ans[reach_times == self.big_num] = 0 # Player never left start and thus failed to select a target
        
        return ans
        
    
    def movement_onset_times(self, task):
        if task == 'reaction':
            if self.metric_type == 'position':
                movement_onset_times = np.argmax(self.raw_data.reaction_dist_data > self.exp_info.start_radius,axis=3)
            elif self.metric_type == 'velocity':
                movement_onset_times = np.argmax(self.raw_data.reaction_speed_data > self.vel_threshold, axis=3)
            elif self.metric_type == 'velocity_linear':
                raise NotImplementedError('Still haven\'t implemented this in the refactor')
        elif task == 'task':
            if self.metric_type == 'position':
                movement_onset_times = np.argmax(self.raw_data.task_dist_data > self.exp_info.start_radius,axis=3)
            elif self.metric_type == 'velocity':
                movement_onset_times = np.argmax(self.raw_data.task_speed_data > self.vel_threshold,axis=3)
            elif self.metric_type == 'velocity_linear':
                raise NotImplementedError('Still haven\'t implemented this in the refactor')
        else:
            raise ValueError('task argument should be \'reaction\' or \'task\'')
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
            ans = self.movement_onset_times(task='reaction')[:,-1,:] #! Last row is the actual reaction, first two are timing for exp1
        elif self.exp_info.experiment == 'Exp2':
            ans = self.movement_onset_times(task='reaction') - self.raw_data.agent_reaction_leave_time
        #* Filter out fault reaction times
        if filter:
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
        ans = np.logical_and(self.task_decision_array!=0, self.raw_data.agent_task_decision_array!=0)
        return ans
    
        
class ScoreMetrics:
    def __init__(self, exp_info: ExperimentInfo, raw_data: RawData, movement_metrics: MovementMetrics):
        self.exp_info = exp_info
        self.raw_data = raw_data
        self.task_decision_array = movement_metrics.task_decision_array
        self.both_reached_mask = movement_metrics.both_reached_mask
        self.agent_task_decision_array = raw_data.agent_task_decision_array
        
    @cached_property
    def win_mask(self):
        ans = np.logical_or(
            self.task_decision_array*self.agent_task_decision_array == 1, 
            np.logical_and(self.task_decision_array!=0, self.agent_task_decision_array==0)
        )
        return ans
    
    @cached_property
    def indecision_mask(self):
        return self.task_decision_array == 0
    
    @cached_property
    def incorrect_mask(self):
        return (self.task_decision_array*self.agent_task_decision_array == -1)
    
    def score_metric(self, score_type):
        '''
        Returns a dictionary of wins, incorrects, and indecisions
        '''
        score_mask_dict = dict(zip(SCORE_METRIC_NAMES,(self.win_mask,self.incorrect_mask,self.indecision_mask)))
        return np.count_nonzero(score_mask_dict[score_type], axis=2)    
    
    @cached_property
    def trial_results(self):
        ans = np.zeros((self.exp_info.num_subjects,self.exp_info.num_task_blocks,self.exp_info.num_task_trials))*np.nan
        ans[self.win_mask] = 1
        ans[self.indecision_mask] = 2
        ans[self.incorrect_mask] = 3  
        return ans
    
    @cached_property
    def exp2_points_scored(self):
        points_c0 = self.score_metric['wins'][0]
        points_c1 = self.score_metric['wins'][1] - self.score_metric['incorrects'][1]
        points_c2 = self.score_metric['wins'][2] - self.score_metric['indecisions'][2] 
        points_c3 = self.score_metric['wins'][3] - self.score_metric['incorrects'][3] - self.score_metric['indecisions'][3] 
        
        return np.array([points_c0,points_c1,points_c2,points_c3])
    
    @cached_property
    def both_reached_mask(self):
        ans = np.logical_and(~self.indecision_mask!=0, 
                                           self.raw_data.agent_task_decision_array!=0)
        return ans
    
    def wins_when_both_reach(self,perc=True):
        win_and_both_reached_mask = self.both_reached_mask*self.win_mask
        if perc:
            # Number of wins when both reached, divided by the number of times both reach 
            return (np.count_nonzero(win_and_both_reached_mask,axis=2)/np.count_nonzero(self.both_reached_mask,axis=2))*100
        else:
            return np.count_nonzero(win_and_both_reached_mask,axis=2)
        
    def incorrects_when_both_reach(self,perc=True):
        incorrect_and_both_reached_mask = self.both_reached_mask*self.incorrect_mask
        if perc:
            # Number of wins when both reached, divided by the number of times both reach 
            return (np.count_nonzero(incorrect_and_both_reached_mask,axis=2)/np.count_nonzero(self.both_reached_mask,axis=2))*100
        else:
            return np.count_nonzero(incorrect_and_both_reached_mask,axis=2)
    
    
class ReactGuessScoreMetrics:
    def __init__(self, exp_info: ExperimentInfo, raw_data: RawData, 
                 movement_metrics: MovementMetrics, score_metrics: ScoreMetrics):
        self.exp_info = exp_info
        self.raw_data = raw_data
        self.reaction_times = movement_metrics.reaction_times
        self.movement_onset_times = movement_metrics.movement_onset_times
        self.incorrect_mask = score_metrics.incorrect_mask
        self.trial_results = score_metrics.trial_results
        self.score_metric = score_metrics.score_metric
        self.both_reached_mask = movement_metrics.both_reached_mask
        self.reaction_time_threshold = 200
        self.react_guess_mask = dict(zip(('react','guess'), self.get_react_guess_mask()))
        
    def set_reaction_time_threshold(self, threshold, **kwargs):
        reaction_time_subtraction = kwargs.get('reaction_time_subtraction')
        if not isinstance(threshold,int):
            self.reaction_time_threshold = np.nanmean(self.reaction_times) - reaction_time_subtraction
        else:
            self.reaction_time_threshold = threshold
    
    def get_react_guess_mask(self):
        guess_mask =  (
            ((self.movement_onset_times(task='task')
            - self.raw_data.agent_task_leave_time) 
            <= self.reaction_time_threshold)
            | self.incorrect_mask
        )
        react_mask = ~guess_mask
        
        return react_mask, guess_mask
    
    def total_reaction_guess(self, react_or_guess):
        return np.count_nonzero(self.react_guess_mask[react_or_guess],axis=2)
    
    def react_guess_results(self, react_or_guess):
        return mask_array(self.trial_results, self.react_guess_mask[react_or_guess])
    
    def react_guess_score_metric_dict(self, react_or_guess):
        wins = np.count_nonzero(self.react_guess_results(react_or_guess)==1,axis=2)
        indecisions = np.count_nonzero(self.react_guess_results(react_or_guess)==2,axis=2)
        incorrects = np.count_nonzero(self.react_guess_results(react_or_guess)==3,axis=2)
        return {'wins':wins, 'incorrects':incorrects, 'indecisions':indecisions}
    
    def react_guess_incorrects(self, react_or_guess):
        return np.count_nonzero(self.react_guess_results(react_or_guess)==2,axis=2)
    
    def react_guess_indecisions(self, react_or_guess):
        return np.count_nonzero(self.react_guess_results(react_or_guess)==3,axis=2)
    
    def react_guess_that_were_score_metric(self, metric, react_or_guess):
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
        denominator = self.score_metric(metric) # Total wins, indecisions, incorrects
        
        return numerator,denominator,np.divide(numerator,denominator, # gamble_wins/total_gambles
                    out=np.zeros_like(numerator)*np.nan,where=denominator!=0)*100
    
    def perc_react_guess_score_metric_when_both_reach(self,metric_name,react_or_guess):
        react_guess_both_reached_mask = self.react_guess_mask[react_or_guess]*self.both_reached_mask
        total_react_guess_when_both_reach = np.count_nonzero(react_guess_both_reached_mask,axis=2)
        react_guess_wins_when_both_decide        = np.count_nonzero(mask_array(self.trial_results,react_guess_both_reached_mask) == 1,axis=2) # Count where the both decided reaction trials are equal to 1 for the trial results array
        react_guess_indecisions_when_both_decide = np.count_nonzero(mask_array(self.trial_results,react_guess_both_reached_mask) == 2,axis=2) # Count where the both decided reaction trials are equal to 1 for the trial results array
        react_guess_incorrects_when_both_decide  = np.count_nonzero(mask_array(self.trial_results,react_guess_both_reached_mask) == 3,axis=2) # Count where the both decided reaction trials are equal to 1 for the trial results array
        perc_wins = np.divide(react_guess_wins_when_both_decide,
                              total_react_guess_when_both_reach, # react_guess_wins_when_both_decide/total_reactions
                              out=np.zeros_like(react_guess_wins_when_both_decide)*np.nan,
                              where=total_react_guess_when_both_reach!=0)*100
        
        perc_indecisions = np.divide(react_guess_indecisions_when_both_decide,
                                     total_react_guess_when_both_reach, # react_guess_wins_when_both_decide/total_reactions
                                     out=np.zeros_like(react_guess_wins_when_both_decide)*np.nan,
                                     where=total_react_guess_when_both_reach!=0)*100
        
        perc_incorrects = np.divide(react_guess_incorrects_when_both_decide,
                                    total_react_guess_when_both_reach, # react_guess_wins_when_both_decide/total_reactions
                                    out=np.zeros_like(react_guess_wins_when_both_decide)*np.nan,
                                    where=total_react_guess_when_both_reach!=0)*100
        ans = dict(zip(SCORE_METRIC_NAMES,(perc_wins,perc_indecisions,perc_incorrects)))
        return ans[metric_name]

class ReactGuessMovementMetrics:
    def __init__(self, exp_info: ExperimentInfo, raw_data: RawData, 
                 movement_metrics: MovementMetrics, score_metrics: ScoreMetrics):
        self.exp_info = exp_info
        self.raw_data = raw_data
        self.reaction_times = movement_metrics.reaction_times
        self.movement_onset_times = movement_metrics.movement_onset_times
        self.incorrect_mask = score_metrics.incorrect_mask
        self.trial_results = score_metrics.trial_results
        self.score_metric = score_metrics.score_metric
        self.both_reached_mask = movement_metrics.both_reached_mask
        self.reaction_time_threshold = 200
        self.react_guess_mask = dict(zip(('react','guess'), self.get_react_guess_mask()))
        
        
    
class SubjectBuilder:
    def __init__(self,subjects,experiment,num_task_blocks,num_task_trials_initial,num_reaction_blocks,
                 num_reaction_trials,num_timing_trials, select_trials,
                 reaction_xypos_data,reaction_dist_data, reaction_xyvelocity_data, 
                 reaction_speed_data, reaction_trial_type_array, reaction_trial_start, 
                 agent_reaction_decision_array, agent_reaction_leave_time, interval_trial_start, interval_reach_time, coincidence_trial_start, coincidence_reach_time, 
                 task_xypos_data, task_dist_data, task_xyvelocity_data, task_speed_data, agent_task_decision_array, agent_task_leave_time,
                 **kwargs):     
        
        self.exp_info = ExperimentInfo(
            subjects = subjects, experiment=experiment, num_task_blocks=num_task_blocks, 
            num_task_trials_initial=num_task_trials_initial, num_reaction_blocks=num_reaction_blocks,
            num_reaction_trials=num_reaction_trials, num_timing_trials=num_timing_trials,
            select_trials=select_trials
        )
        
        self.raw_data = RawData(
            exp_info = self.exp_info,
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
        movement_metrics_type = kwargs.get('movement_metrics_type','velocity')
        self.movement_metrics = MovementMetrics(
            exp_info=self.exp_info, 
            raw_data=self.raw_data, 
            metric_type = movement_metrics_type,
        )
        
        self.score_metrics = ScoreMetrics(
            exp_info=self.exp_info, raw_data = self.raw_data, 
            movement_metrics=self.movement_metrics
        )
        
        self.react_guess_score_metrics = ReactGuessScoreMetrics(
            exp_info=self.exp_info, raw_data=self.raw_data, 
            movement_metrics=self.movement_metrics, 
            score_metrics=self.score_metrics
        )
        
        self.react_guess_movement_metrics = ReactGuessMovementMetrics(
            exp_info=self.exp_info, raw_data=self.raw_data, 
            movement_metrics=self.movement_metrics, 
            score_metrics=self.score_metrics
        )
        
    def __repr__(self):
        return f'{self.__class__.__name__} {self.subject}'
        
        