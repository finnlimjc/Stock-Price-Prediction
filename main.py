#Predicting Stock Prices (Last Update: Oct 8, 2019)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Retrieving Dataset
from pandas_datareader import data
from datetime import datetime
ticker = ['AAPL', 'MSFT', 'SPY']
start_date = '2015-01-02' #Only 2015 onwards due to FREE data source. YYYY-MM-DD
end_date = datetime.now()
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
"""
    Splitting the dataset into 'train' and 'test', under 'train' we have 'x_train' and 'y_train'.
    Removing the same number of values as the length of 'test' under 'shifted_data'.
    'first_ticker' is to make the code more dynamic instead of adjusting the code for every new ticker.
    Calculated 'row_count' in this way to avoid having the length of each individual ticker, and
    get a singular value instead.
"""
splitter = np.random.rand(len(closing_prices)) < 0.8 #Single column splitting.
x_train = closing_prices[splitter]
test = closing_prices[~splitter]
shifted_data = closing_prices.shift(-len(test))
first_ticker = closing_prices.columns[0]
row_count = len(closing_prices) - shifted_data[first_ticker].count()
shifted_array = np.array(shifted_data)
y_train = shifted_array[-row_count:]
y_train = pd.DataFrame(data = {'AAPL': y_train[:, 0], 'MSFT': y_train[:, 1],
                                'SPY': y_train[:, 2]}, index = all_weekdays[-row_count:])

#Benchmark Accuracy
from sklearn.linear_model import LinearRegression
linear_r = LinearRegression()
linear_r.fit(x_train, y_train)
benchmark = linear_r.predict(test)


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
    https://towardsdatascience.com/predicting-stock-prices-with-python-ec1d0c9bece1
"""