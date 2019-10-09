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

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
closing_prices = scaler.fit_transform(closing_prices)
closing_prices = pd.DataFrame(data = {'AAPL': closing_prices[:, 0], 'MSFT': closing_prices[:, 1],
                                        'SPY': closing_prices[:, 2]}, index = all_weekdays)

#Splitting into train/test


#Benchmark Accuracy
'Logistic Regression first then try SVR'

#Plotting the Original Graph
closing_prices = scaler.inverse_transform(closing_prices)
closing_prices = pd.DataFrame(data = {'AAPL': closing_prices[:, 0], 'MSFT': closing_prices[:, 1],
                                        'SPY': closing_prices[:, 2]}, index = all_weekdays)
plt.plot(closing_prices.index, closing_prices['AAPL'], color = 'blue')
plt.plot(closing_prices.index, closing_prices['MSFT'], color = 'green')
plt.plot(closing_prices.index, closing_prices['SPY'], color = 'yellow')
plt.tick_params(labelsize = '8') #Adjusting the xlabel and ylabel font size.
plt.title("Line Chart for Closing Prices")
plt.xlabel('Date')
plt.ylabel("Closing Price ($)")
plt.legend()
plt.grid()
plt.show()

#References
"""
    https://learndatasci.com/tutorials/python-finance-part-yahoo-finance-api-pandas-matplotlib/
"""