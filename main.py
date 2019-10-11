#Predicting Stock Prices (Last Update: Oct 10, 2019)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

#Simple Functions for Neatness
def round_to_2dp (target):
    result = round(target*100, 2)
    return result

#Retrieving Dataset
from pandas_datareader import data
def get_data(ticker, start_date, end_date, source):
    """
    ticker = ticker symbol
    source = source of stock prices. E.g Google Finance, Yahoo Finance, Stooq, Quandl
    returns the dataset requested.
    """
    ticker = ticker
    start_date = start_date #Only 2015 onwards due to FREE data source. YYYY-MM-DD
    dataset = data.DataReader(ticker, source, start_date, end_date)
    return dataset
ticker = 'AAPL'
start_date = '2015-01-02'
end_date = datetime.date.today()
dataset = get_data(ticker, start_date, end_date, 'stooq') #Main settings to change

#Cleaning Dataset
"""
    There will be missing weekdays due to holidays, here we will get those missing values
    by using the previous day closing prices.
"""
closing_prices = dataset['Close']
all_weekdays = pd.date_range(start = start_date, end = end_date, freq = 'B') #B is Business Days.
closing_prices = closing_prices.reindex(all_weekdays) #To include all missing dates.
closing_prices = closing_prices.fillna(method = 'ffill') #ffill stands for Forward Fill
closing_prices = closing_prices.to_frame() #Convert to Dataframe

#Plotting the Original Graph
"""
plt.plot(all_weekdays, closing_prices, color = 'blue')
plt.tick_params(labelsize = '8') #Adjusting the xlabel and ylabel font size.
plt.title("Line Chart for Closing Prices of " + ticker)
plt.xlabel('Date')
plt.ylabel("Closing Price ($)")
plt.grid()
plt.show()
"""

#Shifting Values Out for Prediction and Comparison by using Past Prices to Reflect Future Prices
forecast_out = 20 #days
closing_prices['Prediction'] = closing_prices.shift(-forecast_out) #Shift down by n days

#X and y Values
X = closing_prices[:-forecast_out].drop(columns = 'Prediction')
y = closing_prices[:-forecast_out].drop(columns = 'Close')

#Splitting into Train/Test for Benchmark
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

#Benchmark Accuracy using Linear Regression
from sklearn.linear_model import LinearRegression
linear_r = LinearRegression()
linear_r.fit(x_train, y_train)
benchmark = linear_r.score(x_test, y_test)
print("Linear Regression Accuracy: " + str(round_to_2dp(benchmark)) + "%")

#Cross Validation using Linear Regression to Evaluate Model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
def cross_validation (estimator, x_train, y_train, x_test, y_test, name):
    """
    estimator = model
    x_train = independent variables from train set.
    y_train = dependent variables from train set.
    x_test = independent variables from test set.
    y_test = dependent variables from test set.
    name = model's name (For printing.)
    returns predicted results from test set.
    """
    accuracies = cross_val_score(estimator = estimator, X = x_train, y = y_train, cv = 20)
    predict_result = cross_val_predict(estimator = estimator, X = x_test, y = y_test, cv = 20)
    predscore = metrics.r2_score(y_true = y_test, y_pred = predict_result)
    print(name + " Average Accuracy: " + str(round_to_2dp(accuracies.mean())) + "% (Train Set)")
    print(name + " Average Accuracy: " + str(round_to_2dp(predscore)) + "% (Test Set)")
    return predict_result
linear_r_test = cross_validation(linear_r, x_train, y_train, x_test, y_test, "Linear Regression")

#Predicting Future Prices by n days using Benchmark
def y_pred(estimator, prepared_dataset):
    """
    estimator = model
    returns future predicted prices.
    """
    forecast_out = 20
    forecast = prepared_dataset[-forecast_out:].drop(columns = 'Prediction')
    y_pred = estimator.predict(forecast)
    return y_pred
def plotting_data (y_pred, prepared_dataset):
    """
    y_pred = future predicted prices
    returns data for plotting of charts (REQUIRES 3 CONTAINERS)
    """
    end_date = datetime.date.today()
    tomorrow = end_date + datetime.timedelta(days = 1)
    forecast_out = 20    
    future_weekdays = pd.date_range(start = tomorrow, periods = forecast_out, freq = 'B')
    future_prices = pd.DataFrame(data = {'Prediction': y_pred.flatten()}, index = future_weekdays)
    final_dataset = prepared_dataset.drop(columns = 'Prediction')
    final_dataset = final_dataset.append(future_prices, sort = True)
    return final_dataset, future_weekdays, future_prices

#All Future Predicted Prices
linear_regression = y_pred(estimator = linear_r, prepared_dataset = closing_prices)
linear_regression, lr_weekdays, lr_prices = plotting_data(linear_regression, closing_prices)

#Plotting the Next n Days of Stock Price
def macro_plot (name, all_weekdays, prepared_dataset, future_weekdays, future_prices, ticker):
    """
    name = name of model
    all_weekdays has been declared above.
    future_weekdays = the weekdays that indexes the prices we have predicted.
    future_prices = the predicted prices
    ticker has been declared above.
    shows a line chart of both old and new prices.
    """
    plt.plot(all_weekdays, prepared_dataset['Close'], color = 'blue') #Previous Prices
    plt.plot(future_weekdays, future_prices, color = 'green') #Future Prices
    plt.tick_params(labelsize = '8') #Adjusting the xlabel and ylabel font size.
    plt.title("Line Chart for Closing Prices of " + ticker)
    plt.suptitle("Linear Regression", size = 12)
    plt.xlabel('Date')
    plt.ylabel("Closing Price ($)")
    plt.grid()
    plt.show()
macro_plot("Linear Regression", all_weekdays, closing_prices, lr_weekdays, lr_prices, ticker)

#Plotting the Prediction Only
plt.plot(lr_weekdays, lr_prices, color = 'green') #Future Prices
plt.tick_params(labelsize = '6') #Adjusting the xlabel and ylabel font size.
plt.title("Line Chart for Predicted Prices of " + ticker)
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


difference_end = end_date + datetime.timedelta(days = forecast_out)
difference_weekdays = pd.date_range(start = start_date, end = difference_end, freq = 'B')
difference_amount = len(final_dataset) - len(difference_weekdays) #Due to exclusion of weekends.
future_end = difference_end + datetime.timedelta(days = difference_amount)
future_weekdays = pd.date_range(start = start_date, end = future_end, freq = 'B')
"""