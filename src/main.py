from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.auto import AutoNHITS, AutoLSTM, AutoTCN, AutoTiDE, AutoTSMixer, AutoPatchTST
from neuralforecast.models import TimeLLM
from numpy import count_nonzero
from neuralforecast import NeuralForecast
from utils.extract_series_llm import extract_series_general
from utils.metrics import evaluate_performance
import argparse
import time
import numpy as np
import os
import utils.constants as cons
from utils.llm_tokenizer import select_llm
from utils.plotter import plot_results, plot_metric
import pandas as pd

def parse_arguments(parser):
    parser.add_argument('--dataset_name', type=str, default='vivli_mdi')
    parser.add_argument('--prediction_horizon', type=int, default=8)
    parser.add_argument('--ts_length', type=int, default=96)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers_loader', type=int, default=18)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--read_plot', type=bool, default=False)
    return parser.parse_args()

def check_sparsity(m_data):
    m_data = np.nan_to_num(m_data, 0)
    sparsity = 1.0 - (count_nonzero(m_data) / float(m_data.size))
    return sparsity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM forecasting')
    args = parse_arguments(parser)


    df, train, test = extract_series_general(dataset_name=args.dataset_name,
                                             n_samples=args.n_samples,
                                             prediction_horizon=args.prediction_horizon,
                                             ts_length=args.ts_length)

    llm_config, llm_model, llm_tokenizer = select_llm(name_llm='gpt')

    start_time = time.time()

    model_list = [
        AutoTCN(h=args.prediction_horizon, config=cons.config_tcn,
                num_samples=args.n_trials, search_alg=HyperOptSearch(random_state_seed=0), cpus=args.num_workers_loader,
                gpus=1),
        AutoLSTM(h=args.prediction_horizon, config=cons.config_lstm,
                 num_samples=args.n_trials, search_alg=HyperOptSearch(random_state_seed=0),
                 cpus=args.num_workers_loader,
                 gpus=1),

        AutoNHITS(h=args.prediction_horizon, config=cons.config_nhits,
                  num_samples=args.n_trials, search_alg=HyperOptSearch(random_state_seed=0),
                  cpus=args.num_workers_loader,
                  gpus=1),
        AutoTiDE(h=args.prediction_horizon, config=cons.config_tide,
                 num_samples=args.n_trials, search_alg=HyperOptSearch(random_state_seed=0),
                 cpus=args.num_workers_loader,
                 gpus=1),

        AutoTSMixer(h=args.prediction_horizon, n_series=1, config=cons.config_tsmixer,
                    num_samples=args.n_trials, search_alg=HyperOptSearch(random_state_seed=0),
                    cpus=args.num_workers_loader,
                    gpus=1),
        AutoPatchTST(h=args.prediction_horizon, config=cons.config_patchtst,
                     num_samples=args.n_trials, search_alg=HyperOptSearch(random_state_seed=0),
                     cpus=args.num_workers_loader,
                     gpus=1),
        TimeLLM(h=args.prediction_horizon,
                input_size=args.input_seq_len,
                llm=llm_model,
                llm_config=llm_config,
                llm_tokenizer=llm_tokenizer,
                prompt_prefix ="GPT_llm",
                batch_size=args.batch_size,
                windows_batch_size=args.windows_batch_size,
                random_seed=args.seed,
                num_workers_loader=args.num_workers_loader,
                max_steps=args.max_steps,
                )

    ]

    if  os.listdir(cons.PATH_PROJECT_MODELS):

        nf = NeuralForecast.load(path=cons.PATH_PROJECT_MODELS)

        nf.fit(train[-96:], val_size=0, distributed_config=None, time_col="time", target_col="cgm")

        forecasts = nf.predict(df=train[-96:])
        forecasts.columns = ['Time','CGM_prediction']
        forecasts = forecasts.reset_index(drop = True)

        forecasts.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS, f'{str(forecasts.Time[0])}_forecasting.csv'))

    else:

        # TODO add local scaler param 'standard', 'robust', 'robust-iqr', 'minmax' or 'boxcox'
        nf = NeuralForecast(
            models=model_list,
            freq='15min'
        )

        cv_df = nf.cross_validation(
            df=train[0:300],
            id_col="unique_id",
            time_col="time",
            target_col="cgm",
            verbose=True,
            n_windows=3,
            step_size=1)
        # columns_list=['AutoNHITS', 'AutoTiDE', 'AutoTSMixer', 'AutoPatchTST','cgm', 'unique_id']
        columns_list = ['AutoTCN', 'AutoLSTM', 'AutoNHITS', 'AutoTiDE', 'AutoTSMixer', 'AutoPatchTST', 'TimeLLM',
                        'cgm', 'unique_id']

        losses_val, aggregated_losses_val, best_model_per_patient_val = evaluate_performance(cv_df, columns_list)

        losses_val = losses_val.drop(['unique_id'], axis=1)
        losses_val.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS, 'validation_error_forecasting.csv'))
        position = columns_list.index(best_model_per_patient_val.model[0])

        nf.save(path=cons.PATH_PROJECT_MODELS,
                model_index=[position],
                overwrite=True,
                save_dataset=True)

