import os
from datetime import datetime
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from app.data_utils import download_data, get_dataframe, add_trading_indicators_to_df
from app.environment import get_env
from app.models_utils import evaluate_model
from app.visualization import plot_price

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

# Train data
data_path = download_data(DATA_DIR_PATH, 'BTC/USD', '1h', datetime(2021, 1, 1), datetime(2023, 1, 1))
orig_df = get_dataframe(data_path)
train_df = add_trading_indicators_to_df(orig_df)
plot_price(train_df, 'Train data')

# Test data №1
data_path = download_data(DATA_DIR_PATH, 'BTC/USD', '1h', datetime(2024, 2, 1), datetime(2024, 3, 1))
orig_df = get_dataframe(data_path)
test_df1 = add_trading_indicators_to_df(orig_df)
plot_price(test_df1, 'Test data (increasing)')

# Test data №2
data_path = download_data(DATA_DIR_PATH, 'BTC/USD', '1h', datetime(2023, 8, 1), datetime(2023, 9, 1))
orig_df = get_dataframe(data_path)
test_df2 = add_trading_indicators_to_df(orig_df)
plot_price(test_df2, 'Test data (decreasing)')

# Test data №3
data_path = download_data(DATA_DIR_PATH, 'BTC/USD', '1h', datetime(2023, 9, 1), datetime(2023, 10, 1))
orig_df = get_dataframe(data_path)
test_df3 = add_trading_indicators_to_df(orig_df)
plot_price(test_df3, 'Test data (flat)')

# Test data №4
data_path = download_data(DATA_DIR_PATH, 'BTC/USD', '1h', datetime(2023, 1, 1), datetime(2024, 1, 1))
orig_df = get_dataframe(data_path)
test_df4 = add_trading_indicators_to_df(orig_df)
plot_price(test_df4, 'Test data (long-term)')

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

env_parameters = {
    'verbose': False, 'windows': 100, 'positions': [-1, 0, 1]
}

train_env = get_env('TrainEnv', train_df, env_params={'reward_function': rew_func_sigm2}, **env_parameters)

test_env1 = get_env('TestEnv1', test_df1, env_params={'reward_function': rew_func_sigm2}, **env_parameters)

test_env2 = get_env('TestEnv2', test_df2, env_params={'reward_function': rew_func_sigm2}, **env_parameters)

test_env3 = get_env('TestEnv3', test_df3, env_params={'reward_function': rew_func_sigm2}, **env_parameters)

test_env4 = get_env('TestEnv4', test_df4, env_params={'reward_function': rew_func_sigm2}, **env_parameters)

"""
Models
"""

models = dict()

for model_file_name in os.listdir(MODELS_WEIGHTS_DIR_PATH):
    model_path = os.path.join(MODELS_WEIGHTS_DIR_PATH, model_file_name)
    models[model_file_name.rstrip('.zip')] = (
        PPO.load(model_path)) if model_file_name.startswith('PPO') else RecurrentPPO.load(model_path)
models['B&H'] = 'bh'
models['Random actions'] = 'random'

"""
Evaluation
"""

for model in models.values():
    evaluate_model(model, train_env, display_plots=True)
    evaluate_model(model, test_env1, display_plots=True)
    evaluate_model(model, test_env2, display_plots=True)
    evaluate_model(model, test_env3, display_plots=True)
    evaluate_model(model, test_env4, display_plots=True)
