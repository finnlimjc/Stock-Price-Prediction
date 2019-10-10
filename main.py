#Predicting Stock Prices (Last Update: Oct 10, 2019)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

#Retrieving Dataset
from pandas_datareader import data
ticker = 'AAPL'
start_date = '2015-01-02' #Only 2015 onwards due to FREE data source. YYYY-MM-DD
end_date = datetime.date.today()
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

#Cross Validation using Linear Regression to Evaluate Model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

def cross_validation (estimator, x_train, y_train, x_test, y_test):
    accuracies = cross_val_score(estimator = estimator, X = x_train, y = y_train, cv = 20)
    predict = cross_val_predict(estimator = estimator, X = x_test, y = y_test, cv = 20)
    predscore = metrics.r2_score(y_true = y_test, y_pred = predict)
    print("Linear Regression Average Accuracy: " + str(accuracies.mean()) + " (Train Set)")
    print("Linear Regression Average Accuracy: " + str(predscore) + " (Test Set)")
    return predict

linear_r_kfold = LinearRegression()
linear_r_predscore = cross_validation(linear_r_kfold, x_train, y_train, x_test, y_test)

#Printing All Accuracies
print("Linear Regression Accuracy: " + str(benchmark))
print("Linear Regression Average Accuracy: " + str(linear_r_predscore))

#Predicting Future Prices by n days using Benchmark
def y_pred(estimator, prepared_dataset):
    forecast_out = 20
    forecast = prepared_dataset[-forecast_out:].drop(columns = 'Prediction')
    y_pred = estimator.predict(forecast)
    return y_pred

    end_date = datetime.date.today()
    tomorrow = end_date + datetime.timedelta(days = 1)
    forecast_out = 20    
    future_weekdays = pd.date_range(start = tomorrow, periods = forecast_out, freq = 'B')
    future_prices = pd.DataFrame(data = {'Prediction': y_pred.flatten()}, index = future_weekdays)
    final_dataset = prepared_dataset.drop(columns = 'Prediction')
    final_dataset = final_dataset.append(future_prices, sort = True)
    return final_dataset

linear_regression = y_pred(estimator = linear_r_kfold, prepared_dataset = closing_prices)

#Plotting the Next n Days of Stock Price
plt.plot(all_weekdays, closing_prices['Close'], color = 'blue') #Previous Prices
plt.plot(future_weekdays, future_prices, color = 'green') #Future Prices
plt.tick_params(labelsize = '8') #Adjusting the xlabel and ylabel font size.
plt.title("Line Chart for Closing Prices of " + ticker)
plt.xlabel('Date')
plt.ylabel("Closing Price ($)")
plt.grid()
plt.show()

#Plotting the Prediction Only
plt.plot(future_weekdays, future_prices, color = 'green') #Future Prices
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