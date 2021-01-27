from typing import List

import pandas as pd
import yfinance as yf
import os


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
        stock_df = yf.download(stock, start="2015-01-01", end="2020-03-20")

        if is_tech_ind:
            # create technical indicators
            stock_df = stock_df[["Open","Close"]]
            stock_df = stock_df.assign(Diff=stock_df["Close"] - stock_df["Open"])
            stock_df.drop(columns=["Open"],inplace =True)
        else:
            # just get the close price
            stock_df = stock_df[["Close"]]

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

    return df.values


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
