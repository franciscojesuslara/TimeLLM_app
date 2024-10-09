import pandas as pd
import numpy as np
import utils.constants as cons
import os

def read_time_series():
    df =  last_modified(cons.PATH_PROJECT_DATA)
    df.index = pd.DatetimeIndex(df.dateTime)
    df = df['value'].resample("15min", offset='1min').mean().interpolate().to_frame()
    df['time'] = df.index.values
    return df


def last_modified(path):
    last_modified = None
    time_modified = 0
    for name in os.listdir(path):
        path_aux = os.path.join(path, name)

        if os.path.isfile(path_aux):
            time_mod = os.path.getmtime(path_aux)
            # Comparar para encontrar el más reciente
            if time_mod > time_modified:
                time_modified = time_mod
                last_modified = path_aux

    df = pd.read_csv(path_aux)
    return df
