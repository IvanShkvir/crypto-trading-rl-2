import os
from datetime import datetime
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gym_trading_env.environments import basic_reward_function

from app.data_utils import download_data, get_dataframe, basic_df_preprocessing, split_dataframe
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
df = basic_df_preprocessing(orig_df, scaling='pct_change')
train_df, test_df = split_dataframe(df, ratio=0.8)

"""
Reward functions
"""

def rew_func_log15(history):
    x = history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    return np.log(x) / np.log(1.5)


def rew_func_log5(history):
    x = history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    return np.log(x) / np.log(5)


def rew_func_lg10(history):
    x = history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    return np.log10(x)


def rew_func_sigm1(history):
    x = history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    k = 1
    return 2 / (1 + np.e ** (k*(1-x))) - 1


def rew_func_sigm2(history):
    x = history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    k = 2
    return 2 / (1 + np.e ** (k*(1-x))) - 1

"""
Environments
"""

def envs_generator(reward_functions: list):
    env_parameters = {
        'verbose': False, 'windows': 25, 'positions': [-1, 0, 1]
    }

    for rew_func in reward_functions:
        train_env = get_env('TrainEnv', train_df,
                            env_params={'reward_function': rew_func}, **env_parameters)
        train_eval_env = Monitor(train_env)
        test_env = get_env('TestEnv', test_df,
                           env_params={'reward_function': rew_func}, **env_parameters)
        test_eval_env = Monitor(test_env)
        yield train_env, train_eval_env, test_env, test_eval_env


rew_functions = [basic_reward_function, rew_func_log15, rew_func_log5, rew_func_lg10, rew_func_sigm1, rew_func_sigm2]
gen = envs_generator(rew_functions)
envs = [next(gen) for _ in range(len(rew_functions))]

"""
Models training
"""

model_parameters = {
    'device': 'cuda', 'tensorboard_log': TENSORBOARD_LOGS_DIR_PATH, 'verbose': 0
}
reward_names = [
    'basic',
    'log1.5',
    'log5',
    'log10',
    'sigm1',
    'sigm2'
]

n_epochs = 50
for (train_env, train_eval_env, test_env, test_eval_env), reward_name in zip(envs, reward_names):
    model = PPO('MlpPolicy', env=train_env, **model_parameters)
    model_name = f'{model.__class__.__name__}_{n_epochs}epochs_{reward_name}_reward'
    returns_history = train_model(model, train_env, train_eval_env, test_eval_env,
                                  epochs=n_epochs, model_name=model_name)
    plot_train_test(
        returns_history['train'], returns_history['test'],
        title='Total reward over epochs', xlabel='Epoch', ylabel='Reward')
    model.save(f'{MODELS_WEIGHTS_DIR_PATH}/{model_name}')
