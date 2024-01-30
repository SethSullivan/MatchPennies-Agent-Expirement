import numpy as np
import pandas as pd

def long_df_to_multiarray(df, agg_dict, by=["Subject", "Condition"]):
    temp = df.groupby(by=["Subject","Condition"]).agg(agg_dict).reset_index()
    values = []
    for k in agg_dict.keys():
        values.append(temp.pivot(values=k, index=by[0], columns=by[1]).to_numpy())
        
    return np.array(values)