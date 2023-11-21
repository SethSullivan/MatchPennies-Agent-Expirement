import pandas as pd
from Optimal_Stopping_Object import ModelConstructor  
import constants
import dill
import numpy as np

from initializer import InitialThangs
it = InitialThangs(experiment="Exp1")

    
    
def run_models_from_df(df,EXPERIMENT="Exp1",num_timesteps=1800,expected=True,
                       use_agent_behavior_lookup=False) -> dict:
    """Iterate through dataframe rows to generate models

    Args:
        df (pd.DataFrame): DataFrame with model inputs as the columns
        EXPERIMENT (str, optional): Either "Exp1" or "Exp2". Defaults to "Exp1".
        num_timesteps (int, optional): Max number of timesteps model goes out to. Defaults to 1800.
        expected (bool, optional): Use expected or true optimal decision time when calculating metrics. Defaults to True.

    Returns:
        dict: Dictionary of models with keys being the unique model identfier
        and values being the model itself
    """
    models = {}
    for index,row in df.iterrows():
        model  = ModelConstructor(
            experiment=EXPERIMENT,
            num_blocks=row.num_blocks,
            num_timesteps=num_timesteps,
            agent_means=row.agent_means,
            agent_sds=row.agent_sds,
            reaction_time=row.reaction_time,
            movement_time=row.movement_time,
            reaction_sd=row.reaction_sd, 
            movement_sd=row.movement_sd,
            timing_sd=row.timing_sd,
            guess_switch_delay=row.guess_switch_delay, 
            guess_switch_sd=row.guess_switch_sd, 
            electromechanical_delay=row.electromechanical_delay,
            expected=expected,
            win_reward=row.win_reward,
            incorrect_cost=row.incorrect_cost,
            indecision_cost=row.indecision_cost,
            round_num = 20,
            use_agent_behavior_lookup=use_agent_behavior_lookup,
        )
        models.update({row.Model:model})
    return models




def run_model(model_input_dict, player_inputs, expected, use_agent_behavior_lookup,
              num_timesteps=1800,experiment="Exp1",round_num=20):
    model = ModelConstructor(
        experiment=experiment,
        num_blocks=it.num_blocks,
        num_timesteps=num_timesteps,
        agent_means=np.array([model_input_dict["agent_means"], model_input_dict["agent_means"]])[:, :, np.newaxis],
        agent_sds=np.array([model_input_dict["agent_sds"], model_input_dict["agent_sds"]])[:, :, np.newaxis],
        reaction_time=np.array([player_inputs["rt"], player_inputs["rt"]])[:, np.newaxis, np.newaxis],
        movement_time=np.array([player_inputs["mt"], player_inputs["mt"]])[:, np.newaxis, np.newaxis],
        reaction_sd=np.array([player_inputs["rt_sd"], player_inputs["rt_sd"]])[:, np.newaxis, np.newaxis],  
        movement_sd=np.array([player_inputs["mt_sd"], player_inputs["mt_sd"]])[:, np.newaxis, np.newaxis],
        timing_sd=np.array([[player_inputs['timing_sd']]*it.num_blocks, 
                            [player_inputs['timing_sd']]*it.num_blocks])[:, :, np.newaxis],
        guess_switch_delay=np.array([0.0, 0.0])[:, np.newaxis, np.newaxis], # These are being FIT, so copied models can just have them as 0
        guess_switch_sd=np.array([0.0,0.0])[:, np.newaxis, np.newaxis],   
        electromechanical_delay=np.array([50.0, 50.0],dtype=float)[:, np.newaxis, np.newaxis],
        electromechanical_sd = np.array([10.0,10.0],dtype=float)[:, np.newaxis, np.newaxis],
        expected=expected,  
        win_reward=1.0,
        incorrect_cost=0.0,  #! These are applied onto the base reward matrix in Optimal Model object
        indecision_cost=0.0,
        round_num = round_num,
        use_agent_behavior_lookup=use_agent_behavior_lookup,

    )
    return model