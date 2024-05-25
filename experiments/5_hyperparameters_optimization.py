import os
from datetime import datetime
import numpy as np
import itertools

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from app.data_utils import download_data, get_dataframe, add_trading_indicators_to_df, split_dataframe
from app.environment import get_env

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
    return 2 / (1 + np.e ** (k * (1 - x))) - 1

"""
Environment
"""

env_parameters = {
    'verbose': False, 'windows': 100, 'positions': [-1, 0, 1]
}

train_env = get_env('TrainEnv', train_df,
                    env_params={'reward_function': rew_func_sigm2}, **env_parameters)
train_eval_env = Monitor(train_env)
test_env = get_env('TestEnv', test_df,
                   env_params={'reward_function': rew_func_sigm2}, **env_parameters)
test_eval_env = Monitor(test_env)

"""
Grid search function
"""

def grid_search(model_type: str):
    batch_sizes = [64, 128, 256]
    ent_coefs = [0, 0.05, 0.1]
    clip_ranges = [0.1, 0.2, 0.3]

    best_mean_reward = -np.inf
    best_parameters = None

    for batch_size, ent_coef, clip_range in itertools.product(batch_sizes, ent_coefs, clip_ranges):
        model_parameters = {
            'device': 'cuda',
            'tensorboard_log': TENSORBOARD_LOGS_DIR_PATH,
            'verbose': 0,
            'batch_size': batch_size,
            'ent_coef': ent_coef,
            'clip_range': clip_range
        }

        if model_type == 'PPO':
            model = PPO('MlpPolicy', env=train_env, **model_parameters)
        elif model_type == 'RecurrentPPO':
            model = RecurrentPPO('MlpLstmPolicy', env=train_env, **model_parameters)
        else:
            raise ValueError(f'Invalid model type: {model_type}')

        eval_callback = EvalCallback(test_eval_env, best_model_save_path=MODELS_WEIGHTS_DIR_PATH,
                                     log_path=TENSORBOARD_LOGS_DIR_PATH, eval_freq=500,
                                     deterministic=True, render=False)

        model.learn(total_timesteps=200000, callback=eval_callback)

        mean_reward = eval_callback.last_mean_reward
        if mean_reward is not None and mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_parameters = model_parameters

    return best_parameters

"""
Grid search results
"""

parameters_names = ['batch_size', 'ent_coef', 'clip_range']

ppo_parameters = grid_search('PPO')
recurrent_ppo_parameters = grid_search('RecurrentPPO')

print('PPO best hyperparameters:')
print(*[f'{param}: {value}' for param, value in zip(parameters_names, ppo_parameters)])
print('RecurrentPPO best hyperparameters:')
print(*[f'{param}: {value}' for param, value in zip(parameters_names, recurrent_ppo_parameters)])
