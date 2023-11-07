import pandas as pd
from Optimal_Stopping_Object import ModelConstructor  
    
    
    
def run_models_from_df(df,EXPERIMENT="Exp1",num_timesteps=1800,expected=True,) -> dict:
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
            round_num = 20
        )
        models.update({row.Model:model})
    return models