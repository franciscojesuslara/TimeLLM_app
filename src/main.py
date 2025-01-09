from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.auto import AutoNHITS, AutoLSTM, AutoTCN, AutoTiDE, AutoTSMixer, AutoPatchTST
from neuralforecast.models import TimeLLM
from neuralforecast import NeuralForecast
from utils.metrics import evaluate_performance
import os
import utils.constants as cons
from utils.llm_tokenizer import select_llm
from utils.read import read_time_series


def create_prediction_model(user_id):
    prediction_horizon = 12
    num_workers_loader = -1
    n_trials = 50

    time_series = read_time_series(str(user_id))
    time_series['unique_id'] = 0

    llm_config, llm_model, llm_tokenizer = select_llm(name_llm='gpt')
    model_list = [
        AutoTCN(h=prediction_horizon, config=cons.config_tcn,
                num_samples=n_trials, search_alg=HyperOptSearch(random_state_seed=0), cpus=num_workers_loader),
        AutoLSTM(h=prediction_horizon, config=cons.config_lstm,
                 num_samples=n_trials, search_alg=HyperOptSearch(random_state_seed=0),
                 cpus=num_workers_loader),

        AutoNHITS(h=prediction_horizon, config=cons.config_nhits,
                  num_samples=n_trials, search_alg=HyperOptSearch(random_state_seed=0),
                  cpus=num_workers_loader),
        AutoTiDE(h=prediction_horizon, config=cons.config_tide,
                 num_samples=n_trials, search_alg=HyperOptSearch(random_state_seed=0),
                 cpus=num_workers_loader),
        AutoTSMixer(h=prediction_horizon, n_series=1, config=cons.config_tsmixer,
                    num_samples=n_trials, search_alg=HyperOptSearch(random_state_seed=0),
                    cpus=num_workers_loader,),
        AutoPatchTST(h=prediction_horizon, config=cons.config_patchtst,
                     num_samples=n_trials, search_alg=HyperOptSearch(random_state_seed=0),
                     cpus=num_workers_loader),

        TimeLLM(h=prediction_horizon,
                input_size=cons.seq_len,
                llm=llm_model,
                llm_config=llm_config,
                llm_tokenizer=llm_tokenizer,
                prompt_prefix="GPT_llm",
                batch_size=5,
                windows_batch_size=5,
                random_seed=0,
                num_workers_loader=num_workers_loader,
                max_steps=1000,
                )
                ]

    nf = NeuralForecast(
        models=model_list,
        freq='15min')

    cv_df = nf.cross_validation(
        df=time_series,
        id_col="unique_id",
        time_col="dateTime",
        target_col="value",
        verbose=True,
        n_windows=50,
        step_size=1)

    columns_list = ['AutoTCN', 'AutoLSTM', 'AutoNHITS', 'AutoTiDE', 'AutoTSMixer', 'AutoPatchTST','TimeLLM',
                    'value', 'unique_id']

    losses_val, aggregated_losses_val, best_model_per_patient_val = evaluate_performance(cv_df, columns_list)

    losses_val = losses_val.drop(['unique_id'], axis=1)
    losses_val.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS, f'validation_error_forecasting_{str(user_id)}.csv'))
    position = columns_list.index(best_model_per_patient_val.model[0])

    nf.save(path=os.path.join(cons.PATH_PROJECT_MODELS, str(user_id)),
            model_index=[position],
            overwrite=True,
            save_dataset=True)

def get_glucose_prediction(user_id):

    time_series = read_time_series(str(user_id))
    time_series['unique_id'] = 0

    nf = NeuralForecast.load(path=os.path.join(cons.PATH_PROJECT_MODELS, str(user_id)))

    forecasts = nf.predict(df=time_series)
    forecasts.columns = ['Time','value_prediction']
    forecasts = forecasts.reset_index(drop = True)

    forecasts.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS, f'{str(forecasts.Time[0])}_forecasting_{str(user_id)}.csv'))