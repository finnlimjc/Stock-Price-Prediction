#Predicting Stock Prices (Last Update: Oct 8, 2019)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Retrieving Dataset
from pandas_datareader import data
from datetime import datetime
ticker = 'AAPL'
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

#Splitting into train/test
forecast_out = 30 #days
closing_prices['Prediction'] = closing_prices.shift(-forecast_out) #Shift down by n days


#Benchmark Accuracy
from sklearn.linear_model import LinearRegression
linear_r = LinearRegression()
linear_r.fit(x_train, y_train)
benchmark = linear_r.predict(test)


#Plotting the Original Graph
plt.plot(closing_prices.index, closing_prices, color = 'blue')
plt.tick_params(labelsize = '8') #Adjusting the xlabel and ylabel font size.
plt.title("Line Chart for Closing Prices of " + ticker)
plt.xlabel('Date')
plt.ylabel("Closing Price ($)")
plt.grid()
plt.show()

#References
"""
    https://learndatasci.com/tutorials/python-finance-part-yahoo-finance-api-pandas-matplotlib/
    https://towardsdatascience.com/predicting-stock-prices-with-python-ec1d0c9bece1
    https://medium.com/@randerson112358/predict-stock-prices-using-python-machine-learning-53aa024da20a
"""

#Various Attempts
"""
    Splitting the dataset into 'train' and 'test', under 'train' we have 'x_train' and 'y_train'.
    Removing the same number of values as the length of 'test' under 'shifted_data'.
    'first_ticker' is to make the code more dynamic instead of adjusting the code for every new ticker.
    Calculated 'row_count' in this way to avoid having the length of each individual ticker, and
    get a singular value instead.

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
"""