import matplotlib.pyplot as plt
import numpy as np
from darts import TimeSeries
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy.spatial import distance
import utils.constants as cons
import os
import pandas as pd
from datasetsforecast.losses import mse, mae, rmse, rmae

def compute_losses(group):
    actual = group['value']
    predicted = group['predicted']
    mse_loss = mse(actual, predicted)
    mae_loss = mae(actual, predicted)
    rmse_loss = rmse(actual, predicted)
    return pd.Series({
        'mse': mse_loss,
        'mae': mae_loss,
        'rmse': rmse_loss,
    })
def evaluate_performance(df, columnas, df_val=None):
    df = df.reset_index()[columnas]

    melted_df = df.melt(id_vars=['unique_id', 'value'], var_name='model', value_name='predicted')
    losses = melted_df.groupby(['unique_id', 'model']).apply(compute_losses).reset_index()
    aggregated_losses = losses.groupby('model').agg(
        mse_mean=('mse', 'mean'),
        mse_std=('mse', 'std'),
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'),
        rmse_std=('rmse', 'std'),
    ).reset_index()
    if df_val is not None:

        df_val = df_val[df_val['unique_id'] != '1170_0']
        df_val = df_val[df_val['unique_id'] != '1271_14']
        df_val = df_val[df_val['unique_id'] != '1711_1']

        df_val = df_val[df_val['unique_id'] != '1536_1']
        df_val = df_val[df_val['unique_id'] != '870_1']
        df_val = df_val[df_val['unique_id'] != '1611_4']
        df_val = df_val[df_val['unique_id'] != '1622_1']
        best_model_per_patient = pd.merge(losses, df_val, on=['unique_id', 'model'],suffixes=('_test', '_val'),
                                          how='inner')
    else:
        best_model_per_patient = losses.loc[losses.groupby('unique_id')['mae'].idxmin()].reset_index(drop=True)
    print(best_model_per_patient['model'].value_counts())
    return losses, aggregated_losses, best_model_per_patient

