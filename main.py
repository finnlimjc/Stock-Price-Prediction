#Predicting Stock Prices (Last Update: Oct 8, 2019)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Retrieving Dataset
from pandas_datareader import data
ticker = ['AAPL', 'MSFT', 'SPY']
start_date = '2015-01-02' #Only 2015 onwards due to FREE data source. YYYY-MM-DD
end_date = '2019-01-01'
dataset = data.DataReader(ticker, 'stooq', start_date, end_date)

#Cleaning Dataset
"""
    There will be missing weekdays due to holidays, here we will get those missing values
    by using the previous day closing prices.
"""
closing_prices = dataset['Close']
all_weekdays = pd.date_range(start = start_date, end = end_date, freq = 'B') #B is Business Days.
closing_prices = closing_prices.reindex(all_weekdays) #To include all missing dates.
closing_prices = closing_prices.fillna(method = 'ffill') #ffill stands for Forward Fill


#References
"""
    https://learndatasci.com/tutorials/python-finance-part-yahoo-finance-api-pandas-matplotlib/