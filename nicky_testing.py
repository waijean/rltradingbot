

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import datetime
import os
from yahoodownloader import YahooDownloader
import sys


gcp_hack = ['aapl', 'msi', 'sbux']

data_df = YahooDownloader(start_date = '2015-01-01',
                          end_date = '2021-12-01',
                          ticker_list = gcp_hack).fetch_data()
data_df.head()

