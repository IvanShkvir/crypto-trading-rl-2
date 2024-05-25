import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd


def plot_price(ohlcv: pd.DataFrame, title: str, **kwargs):
    mpf.plot(ohlcv, title=title, type=kwargs.get('type', 'line'), style=kwargs.get('style', 'binance'),
             ylabel='Price (USDT)', volume='volume' in ohlcv)


def plot_train_test(train: list[float], test: list[float], **kwargs):
    assert len(train) == len(test)
    epochs = len(train)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train, marker='o', linestyle='-', color='b', label='Training Data')
    plt.plot(range(1, epochs+1), test, marker='o', linestyle='-', color='r', label='Test Data')
    if title := kwargs.get('title', None):
        plt.title(title)
    if xlabel := kwargs.get('xlabel', None):
        plt.xlabel(xlabel)
    if ylabel := kwargs.get('ylabel', None):
        plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
