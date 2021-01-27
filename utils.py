import pandas as pd
import os


def get_data():
    """
    Return n x (s+t) ndarray where
    - n is the number of days,
    - s is the number of stocks
    - t is the number of technical indicators

    The three stocks we are using is
    0 = AAPL
    1 = MSI
    2 = SBUX
    """
    df = pd.read_csv("./aapl_msi_sbux.csv")
    # TODO add columns of technical indicators
    return df.values


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
