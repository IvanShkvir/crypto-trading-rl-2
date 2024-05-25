from time import time
from collections import defaultdict
import numpy as np
import pandas as pd

import warnings; warnings.filterwarnings("ignore", category=DeprecationWarning)

import quantstats_lumi as qs
from gym_trading_env.environments import TradingEnv

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm


def collect_model_predictions(model: OnPolicyAlgorithm, env: TradingEnv):
    done, truncated = False, False
    observation, info = env.reset()
    state = None
    episode_starts = np.ones((1,), dtype=bool)
    history = defaultdict(list)

    while not done and not truncated:
        if model == 'bh':
            action = 2
        elif model == 'random':
            action = env.action_space.sample()
        else:
            action, state = model.predict(observation, state=state, episode_start=episode_starts)
        observation, reward, done, truncated, info = env.step(action)
        episode_starts = done
        for k, v in info.items():
            history[k].append(v)
    return history


def train_model(model: OnPolicyAlgorithm, train_env: TradingEnv, train_eval_env: TradingEnv, test_eval_env: TradingEnv,
                epochs: int, **kwargs):
    train_env_timestamps = len(train_env.unwrapped.df)
    model_name = kwargs.get('model_name', f'{model.__class__.__name__}_{epochs}epochs')

    st = time()
    history = defaultdict(list)
    for i in range(1, epochs + 1):
        train_env.reset()
        print(f'[{i}/{epochs}] Training...', end=' ')
        model.learn(total_timesteps=train_env_timestamps, tb_log_name=model_name,
                    callback=TensorboardCallback(), reset_num_timesteps=False)

        print('Evaluating on train...', end=' ')
        train_mean_reward, _ = evaluate_policy(model, train_eval_env, kwargs.get('eval_episodes', 1))
        print('Evaluating on test...', end=' ')
        test_mean_reward, _ = evaluate_policy(model, test_eval_env, kwargs.get('eval_episodes', 1))

        history['train'].append(train_mean_reward)
        history['test'].append(test_mean_reward)

        print('Results...')
        print(f'\tTrain: {train_mean_reward:.3f} | Test: {test_mean_reward:.3f} | ', end='')
        time_in_minutes = (time() - st) / 60
        print(f'Time: {time_in_minutes:.2f} minutes')

    return history


def evaluate_model(model: OnPolicyAlgorithm, env: TradingEnv, display_plots: bool = False, save_to_file: str = None,
                   save_for_render: str = None):
    history = collect_model_predictions(model, env)
    if save_for_render:
        env.unwrapped.save_for_render(dir=save_for_render)

    returns = pd.Series(history['portfolio_valuation'], index=history['date']).pct_change()
    if display_plots:
        qs.reports.full(returns, warn_singular=False)
    else:
        qs.reports.metrics(returns)
    if save_to_file:
        qs.reports.html(returns=returns, output=f'{save_to_file}.html')


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.training_env.buf_infos[-1]['reward']
        position = self.training_env.buf_infos[-1]['position']
        self.logger.record("train/reward", reward)
        self.logger.record("train/position", position)
        return True
