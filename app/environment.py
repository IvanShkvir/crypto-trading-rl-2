import pandas as pd

import gymnasium as gym
import gym_trading_env
from gym_trading_env.environments import basic_reward_function


def get_env(name: str, df: pd.DataFrame, env_params: dict = None, **kwargs):
    _env_params = {
        'positions': [-1, 0, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        'trading_fees': 0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
        'borrow_interest_rate': 0.0003 / 100,  # 0.0003% per timestep (one timestep = 1h here)
        'reward_function': basic_reward_function,
    }
    if env_params:
        _env_params.update(env_params)

    env = gym.make(
        'TradingEnv',
        name=name,
        df=df,
        **_env_params,
        **kwargs
    )

    return env
