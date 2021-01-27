

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import datetime
import os
from yahoodownloader import YahooDownloader
import sys

test = np.load('linear_rl_trader_rewards/test.npy')
import pandas as pd
test = pd.DataFrame(test)

from stockstats import StockDataFrame as Sdf

gcp_hack = ['aapl', 'msi', 'sbux']

data_df = YahooDownloader(start_date = '2015-01-01',
                          end_date = '2021-12-01',
                          ticker_list = gcp_hack).fetch_data()
data_df.head()

tech_indicator_list = ['macd', 'rsi_30', 'cci_30', 'dx_30']

def add_technical_indicator(data, tech_indicator_list):
        """
        calcualte technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df[indicator] = indicator_df
        return df

data_t = add_technical_indicator(data_df, tech_indicator_list)