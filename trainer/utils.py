import pandas as pd
import os


def get_data():
    # returns a T x 3 list of stock prices
    # each row is a different stock
    # 0 = AAPL
    # 1 = MSI
    # 2 = SBUX
    # Let's use AAPL (Apple), MSI (Motorola), SBUX (Starbucks)
    df = pd.read_csv("../aapl_msi_sbux.csv")
    return df.values


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
