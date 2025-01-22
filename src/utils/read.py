import pandas as pd
import numpy as np
import utils.constants as cons
import os
import warnings

def read_time_series(user_id):
    df =  last_modified(cons.PATH_PROJECT_DATA, user_id)
    df.index = pd.DatetimeIndex(df.dateTime)
    df = df['value'].resample("15min", offset='1min').mean()
    nan_values = df.isna().sum()
    if  nan_values/len(df) > 0.25:
        warnings.warn("Forecasting may not be reliable due to a high percentage of missing data.")

    df = df.interpolate().to_frame()
    df = df.reset_index()
    return df


def last_modified(path,user_id):
    last_modified = None
    time_modified = 0
    for name in os.listdir(path):
        if user_id in name:
            path_aux = os.path.join(path, name)
            if os.path.isfile(path_aux):
                time_mod = os.path.getmtime(path_aux)
                # Comparar para encontrar el más reciente
                if time_mod > time_modified:
                    time_modified = time_mod
                    last_modified = path_aux

    df = pd.read_csv(last_modified)
    return df
