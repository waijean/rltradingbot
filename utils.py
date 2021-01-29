from typing import List

import pandas as pd
import yfinance as yf
import os
from stockstats import StockDataFrame


def get_data_from_yf(stocks:List[str], is_tech_ind:bool):

    """
    Download stock prices from yahoo finance API

    Args:
        stocks: List of stocks. For example, ["AAPL","MSI","SBUX"]

    Returns: n x (s+t) ndarray where
    - n is the number of days
    - s is the number of stocks
    - t is the number of technical indicators

    """
    df = pd.DataFrame()
    for stock in stocks:  # :
        stock_df = yf.download(stock, start="2013-02-01", end="2017-02-01")
        stock_sdf = StockDataFrame.retype(stock_df)
        if is_tech_ind:
            # create technical indicators
            stock_df['rsi_10'] = pd.DataFrame(stock_sdf['rsi_10'], index=stock_df.index)
            stock_df['macd'] = pd.DataFrame(stock_sdf['macd'], index=stock_df.index)
            stock_df['wr_10'] = pd.DataFrame(stock_sdf['wr_10'], index=stock_df.index)
            stock_df = stock_df[['close', 'rsi_10', 'macd', 'wr_10']]
        else:
            # just get the close price
            stock_df = stock_df[["close"]]

        # add stock prefix to column names
        colnames = [col + "_" + stock for col in stock_df.columns]
        stock_df.columns = colnames

        df = pd.concat([df, stock_df], axis=1)

    # make sure there are no null values
    assert all(df.notna())

    # sort date
    df.sort_index(inplace=True)

    # sort columns (a quick hack so the Close Price will be the first s columns and technical indicators will be last)
    df = df[sorted(df.columns)]
    df = df.iloc[1:,]

    return df.to_numpy()

def get_data():
    """
    Returns: n x s ndarray where
    - n is the number of days
    - s is the number of stocks

    The three stocks we are using is
    0 = AAPL
    1 = MSI
    2 = SBUX
    """
    df = pd.read_csv("./aapl_msi_sbux.csv")
    return df.values


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
