import os
from datetime import datetime

import pandas as pd
import pandas_ta as ta

from gym_trading_env import downloader
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def download_data(data_dir: str, pair: str, timeframe: str,
                  since: datetime, until: datetime):
    download_path = os.path.join(data_dir, f'{since.date()}__{until.date()}')
    download_file = f'binance-{pair.replace("/", "")}-{timeframe}.pkl'
    if not os.path.exists(os.path.join(download_path, download_file)):
        os.makedirs(download_path, exist_ok=True)
        downloader.download(
            exchange_names=['binance'],
            symbols=[pair],
            timeframe=timeframe,
            dir=download_path,
            since=since,
            until=until
        )

    return os.path.join(download_path, download_file)


def get_dataframe(path: str):
    df = pd.read_pickle(path)
    return df


def split_dataframe(df: pd.DataFrame, ratio: float):
    split_index = int(len(df) * ratio)

    df1 = df[:split_index]
    df2 = df[split_index:]

    return df1, df2


def basic_df_preprocessing(df: pd.DataFrame, scaling: str):
    if scaling == 'std':
        scaler = StandardScaler()
        df['feature_close'] = scaler.fit_transform(df['close'].values.reshape(-1, 1))
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
        df['feature_close'] = scaler.fit_transform(df['close'].values.reshape(-1, 1))
    elif scaling == 'pct_change':
        df['feature_close'] = df['close'].pct_change()
    else:
        raise ValueError(f'Unknown scaling method {scaling}')

    df['feature_close'] = df['feature_close'].shift(1)
    df = df.iloc[1+1:]  # one is used because of the data shift to know only the previous close price
                        # and one is used because of the additional data shift applied in .pct_change()
    return df


def add_trading_indicators_to_df(df: pd.DataFrame):
    indicators_df = df.copy()

    indicators_df['MA_20'] = ta.sma(indicators_df['close'], length=20)
    indicators_df['EMA_12'] = ta.ema(indicators_df['close'], length=12)
    indicators_df['EMA_26'] = ta.ema(indicators_df['close'], length=26)
    indicators_df[['%K', '%D']] = ta.stoch(indicators_df['high'], indicators_df['low'], indicators_df['close'])
    indicators_df[['BBL_20', 'BBM_20']] = ta.bbands(indicators_df['close'], length=20)[['BBL_20_2.0', 'BBM_20_2.0']]
    indicators_df['RSI_14'] = ta.rsi(indicators_df['close'], length=14)
    indicators_df[['senkou_span_a', 'senkou_span_b']] = ta.ichimoku(indicators_df['high'], indicators_df['low'], indicators_df['close'])[0][['ITS_9', 'IKS_26']]
    indicators_df['SMA_20'] = ta.sma(indicators_df['close'], length=20)
    indicators_df['WMA_20'] = ta.wma(indicators_df['close'], length=20)
    indicators_df['WMA_50'] = ta.wma(indicators_df['close'], length=50)

    indicators_df.ffill(inplace=True)

    for column in indicators_df.drop('date_close', axis=1).columns:
        df[f'feature_{column}'] = indicators_df[column].shift(1)

    for column in [col for col in list(df.columns) if col.startswith('feature_')]:
        df[column] = df[column].pct_change()

    df.dropna(axis=0, inplace=True)

    return df
