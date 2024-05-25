import os
from datetime import datetime
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from app.data_utils import download_data, get_dataframe, add_trading_indicators_to_df, split_dataframe
from app.environment import get_env
from app.models_utils import train_model
from app.visualization import plot_train_test

"""
Constants
"""

PROJ_PATH = os.path.abspath('')
DATA_DIR_PATH = os.path.join(PROJ_PATH, 'data')
TENSORBOARD_LOGS_DIR_PATH = os.path.join(PROJ_PATH, 'tb_logs')
MODELS_WEIGHTS_DIR_PATH = os.path.join(PROJ_PATH, 'models')

"""
Data downloading and preprocessing
"""

data_path = download_data(DATA_DIR_PATH, 'BTC/USD', '1h', datetime(2021, 1, 1), datetime(2023, 1, 1))
orig_df = get_dataframe(data_path)
df = add_trading_indicators_to_df(orig_df)
train_df, test_df = split_dataframe(df, ratio=0.8)

"""
Reward function
"""

def rew_func_sigm2(history):
    x = history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    k = 2
    return 2 / (1 + np.e ** (k*(1-x))) - 1

"""
Environments
"""

def envs_generator(window_sizes: list):
    env_parameters = {
        'verbose': False, 'positions': [-1, 0, 1]
    }

    for ws in window_sizes:
        train_env = get_env('TrainEnv', train_df,
                            env_params={'reward_function': rew_func_sigm2}, **{**env_parameters, 'windows': ws})
        train_eval_env = Monitor(train_env)
        test_env = get_env('TestEnv', test_df,
                           env_params={'reward_function': rew_func_sigm2}, **{**env_parameters, 'windows': ws})
        test_eval_env = Monitor(test_env)
        yield train_env, train_eval_env, test_env, test_eval_env


window_sizes = [5, 25, 50, 100]
gen = envs_generator(window_sizes)
envs = [next(gen) for _ in range(len(window_sizes))]

"""
Models training
"""

model_parameters = {
    'device': 'cuda', 'tensorboard_log': TENSORBOARD_LOGS_DIR_PATH, 'verbose': 0
}

n_epochs = 50
for (train_env, train_eval_env, test_env, test_eval_env), ws in zip(envs, window_sizes):
    model = PPO('MlpPolicy', env=train_env, **model_parameters)
    model_name = f'{model.__class__.__name__}_{n_epochs}epochs_ws_{ws}'
    returns_history = train_model(model, train_env, train_eval_env, test_eval_env,
                                  epochs=n_epochs, model_name=model_name)
    plot_train_test(
        returns_history['train'], returns_history['test'],
        title='Total reward over epochs', xlabel='Epoch', ylabel='Reward')
    model.save(f'{MODELS_WEIGHTS_DIR_PATH}/{model_name}')
