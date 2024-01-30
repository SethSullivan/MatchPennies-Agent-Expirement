import dill 
import numpy as np
import numba as nb
from scipy import stats
import numba_scipy  # Needs to be imported so that numba recognizes scipy (specificall scipy special erf)
from numba_stats import norm

from copy import deepcopy
from tqdm import tqdm

import read_data_functions as rdf
import constants
from initializer import InitialThangs
from Optimal_Stopping_Object import ModelConstructor, get_moments, nb_sum

it = InitialThangs("Exp1")
#* Pickle model inputs
group = rdf.generate_subject_object_v3('Exp1', "All Trials")
rt = np.nanmedian(group.movement_metrics.reaction_times, axis=1) - 25
rt_sd = np.nanstd(group.movement_metrics.reaction_times, axis=1)
mt = np.min(np.nanmedian(group.movement_metrics.movement_times("task"), axis=2), axis=1)
mt_sd = np.min(np.nanstd(group.movement_metrics.movement_times("task"), axis=2), axis=1)
timing_sd = np.nanstd(group.movement_metrics.coincidence_reach_time, axis=1) #! This nee
agent_sds = np.nanmedian(np.nanstd(group.raw_data.agent_task_leave_time, axis=2),axis=0)
agent_means = np.nanmedian(np.nanmean(group.raw_data.agent_task_leave_time, axis=2),axis=0)
metrics = [rt,rt_sd,mt,mt_sd,timing_sd,agent_sds,agent_means]
keys = ["rt","rt_sd","mt","mt_sd","timing_sd","agent_sds","agent_means"]
model_input_dict = dict(zip(keys,metrics))
with open(constants.MODEL_INPUT_PATH / 'model_input_dict.pkl','wb') as f:
    dill.dump(model_input_dict,f)

#* Pickle true median data for loss function
participant_median_movement_onset_time = np.nanmedian(group.movement_metrics.movement_onset_times("task"), axis=2)
with open(constants.MODEL_INPUT_PATH / 'participant_median_movement_onset_time.pkl','wb') as f:
    dill.dump(participant_median_movement_onset_time,f)
participant_mean_movement_onset_time = np.nanmean(group.movement_metrics.movement_onset_times("task"), axis=2)
with open(constants.MODEL_INPUT_PATH / 'participant_mean_movement_onset_time.pkl','wb') as f:
    dill.dump(participant_mean_movement_onset_time,f)
participant_sd_movement_onset_time = np.nanstd(group.movement_metrics.movement_onset_times("task"), axis=2)
with open(constants.MODEL_INPUT_PATH / 'participant_sd_movement_onset_time.pkl','wb') as f:
    dill.dump(participant_sd_movement_onset_time,f)
wins = group.score_metrics.score_metric("wins")
with open(constants.MODEL_INPUT_PATH / 'participant_wins.pkl','wb') as f:
    dill.dump(wins,f)
incorrects = group.score_metrics.score_metric("incorrects")
with open(constants.MODEL_INPUT_PATH / 'participant_incorrects.pkl','wb') as f:
    dill.dump(incorrects,f)
indecisions = group.score_metrics.score_metric("indecisions")
with open(constants.MODEL_INPUT_PATH / 'participant_indecisions.pkl','wb') as f:
    dill.dump(indecisions,f)


#* Bootstrap parameters
if False:
    @nb.njit(parallel=True)
    def parameter_bootstrap(parameters:np.ndarray,M=1e4,):
        print('begin bootstrap')
        M = int(M)
        num_params = len(parameters)
        num_subjects = len(parameters[0])
        distribution = np.zeros((M,num_params))*np.nan
        results = np.zeros((M,num_params))*np.nan
        participant_ids = np.zeros((M,num_subjects),dtype=np.int32)
        for i in nb.prange(M):
            # One set of participant ids, use across every paramter
            participant_ids[i,:] = np.random.randint(0,num_subjects,size=num_subjects)
            
            # Use those ids to get all parameters
            for j in range(num_params):
                distribution[i,j] = np.nanmean(parameters[j][participant_ids[i,:]])
                
            results[i,:] = np.mean(distribution[i,:])
            
        return distribution, results, participant_ids

    bootstrap_parameters = np.array([x for x in model_input_dict.values() if x.shape[0]==it.num_subjects])
    a,b,c = parameter_bootstrap(parameters=bootstrap_parameters, M=1e2) # initialize bootstrap

    parameter_distribution,results,participant_ids = parameter_bootstrap(parameters=bootstrap_parameters, )
    print("finished bootstrap")
    with open(constants.MODEL_INPUT_PATH / 'bootstrap_parameter_distribution.pkl','wb') as f:
        dill.dump(parameter_distribution,f)    
    with open(constants.MODEL_INPUT_PATH / 'bootstrap_results.pkl','wb') as f:
        dill.dump(results,f)    
    with open(constants.MODEL_INPUT_PATH / 'participant_ids.pkl','wb') as f:
        dill.dump(participant_ids,f)  
    
    
#* Generate timing_sd lookup table
def agent_moments(timesteps,timing_sd,agent_means,agent_sds,nsteps=1):
        """
        Get first three central moments (EX2 is normalized for mean,
        EX3 is normalized for mean and sd) of the new distribution based on timing uncertainty
        """
        #* Steps done outside for loop in get_moments to make it faster
        #
        # DOing this bc lru cache needs a hashable data type (aka a float or int)
        # We then recreate the timing_sd array which is just one number in (2,6,1) shape            
        # Creates a 1,1,2000 inf timesteps, that can broadcast to 2,6,1
        inf_timesteps = np.arange(0.0, 2000.0, nsteps)[np.newaxis,np.newaxis,:] # Going to 2000 is a good approximation, doesn't get better by going higher
        time_means = timesteps[0,0,:] # Get the timing means that player can select as their stopping time
        agent_pdf = norm.pdf(inf_timesteps, agent_means, agent_sds)  # Find agent pdf
        prob_agent_less_player = norm.cdf(
            0, agent_means - inf_timesteps, 
            np.sqrt(agent_sds**2 + (timing_sd) ** 2)
        )
        #* Returns tuple of First three moments for left (Reaction) and right (Gamble) of distribution
        #* Don't actually use the skew at all
        return_vals = get_moments(
            inf_timesteps.squeeze(), 
            time_means.squeeze(), 
            timing_sd.squeeze(), # Squeezing for numba
            prob_agent_less_player.squeeze(), 
            agent_pdf.squeeze()
        )
        return return_vals
    
def cutoff_agent_behavior(*args):
    # Get the First Three moments for the left and right distributions (if X<Y and if X>Y respectively)
    EX_R, EX2_R, EX_G, EX2_G = agent_moments(*args)
    # no_inf_moments = [np.nan_to_num(x,nan=np.nan,posinf=np.nan,neginf=np.nan) for x in moments]

    reaction_leave_time, reaction_leave_time_var = EX_R, EX2_R
    reaction_leave_time_sd = np.sqrt(reaction_leave_time_var)

    guess_leave_time, guess_leave_time_var, = EX_G, EX2_G
    guess_leave_time_sd = np.sqrt(guess_leave_time_var)
    return reaction_leave_time,reaction_leave_time_sd,guess_leave_time,guess_leave_time_sd

def find_lookups(max_timing_sd = 100, timesteps = np.tile(np.arange(0.0, float(1800), 1), (2, 6, 1))):
    print("starting lookup")
    input_keys = ["rt","rt_sd","mt","mt_sd","timing_sd",]
    true_parameters = [np.nanmedian(v) for k,v in model_input_dict.items() if "agent" not in k] 
    player_inputs = dict(zip(input_keys,true_parameters))

    reaction_leave_time_lookup = np.zeros((max_timing_sd,max_timing_sd,2,6,1800))
    reaction_leave_time_sd_lookup = np.zeros_like(reaction_leave_time_lookup)
    guess_leave_time_lookup = np.zeros_like(reaction_leave_time_lookup)
    guess_leave_time_sd_lookup = np.zeros_like(reaction_leave_time_lookup)

    for i in tqdm(range(1,max_timing_sd)):
        for j in range(1,max_timing_sd):
            known = np.repeat(float(i),6)
            unknown = np.repeat(float(j),6)
            timing_sd = np.vstack((known,unknown))[:, :, np.newaxis]
            
            model = ModelConstructor(
                experiment="Exp1",
                num_blocks=it.num_blocks,
                num_timesteps=1800,
                agent_means=np.array([model_input_dict["agent_means"], model_input_dict["agent_means"]])[:, :, np.newaxis],
                agent_sds=np.array([model_input_dict["agent_sds"], model_input_dict["agent_sds"]])[:, :, np.newaxis],  #!
                reaction_time=np.array([player_inputs["rt"], player_inputs["rt"]])[:, np.newaxis, np.newaxis],
                movement_time=np.array([player_inputs["mt"], player_inputs["mt"]])[:, np.newaxis, np.newaxis],
                reaction_sd=np.array([player_inputs["rt_sd"], player_inputs["rt_sd"]])[:, np.newaxis, np.newaxis],  #! Reducing these, aka the particiapnt thinks they are more certain than they are
                movement_sd=np.array([player_inputs["mt_sd"], player_inputs["mt_sd"]])[:, np.newaxis, np.newaxis],
                timing_sd=timing_sd,
                guess_switch_delay=np.array([0, 0])[:, np.newaxis, np.newaxis], # These are being FIT, so copied models can just have them as 0
                guess_switch_sd=np.array([0,0])[:, np.newaxis, np.newaxis],   
                electromechanical_delay=np.array([50, 50])[:, np.newaxis, np.newaxis],
                electromechanical_sd = np.array([10,10])[:, np.newaxis, np.newaxis],
                switch_cost_exists=True,
                expected=True, 
                win_reward=1.0,
                incorrect_cost=0.0,  #! These are applied onto the base reward matrix in Optimal Model object
                indecision_cost=0.0,
                round_num = 20,
                use_agent_behavior_lookup = True,
            )
            reaction_leave_time_lookup[i,j,:,:] = model.agent_behavior.reaction_leave_time
            reaction_leave_time_sd_lookup[i,j,:,:] = model.agent_behavior.reaction_leave_time_sd
            guess_leave_time_lookup[i,j,:,:] = model.agent_behavior.guess_leave_time
            guess_leave_time_sd_lookup[i,j,:,:] = model.agent_behavior.guess_leave_time_sd
            
    return (reaction_leave_time_lookup,reaction_leave_time_sd_lookup,
            guess_leave_time_lookup,guess_leave_time_sd_lookup)


reaction_leave_time_lookup,reaction_leave_time_sd_lookup,\
        guess_leave_time_lookup,guess_leave_time_sd_lookup = find_lookups()
with open(constants.MODEL_INPUT_PATH / 'reaction_leave_time_lookup.pkl','wb') as f:
        dill.dump(reaction_leave_time_lookup,f)
with open(constants.MODEL_INPUT_PATH / 'reaction_leave_time_sd_lookup.pkl','wb') as f:
    dill.dump(reaction_leave_time_sd_lookup,f)
with open(constants.MODEL_INPUT_PATH / 'guess_leave_time_lookup.pkl','wb') as f:
    dill.dump(guess_leave_time_lookup,f)
with open(constants.MODEL_INPUT_PATH / 'guess_leave_time_sd_lookup.pkl','wb') as f:
    dill.dump(guess_leave_time_sd_lookup,f)