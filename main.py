#Predicting Stock Prices (Last Update: Oct 12, 2019)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import xgboost as xgb

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
    if (dataset.empty is True):
        print("Ticker symbol, source or dates are either unavailable or incorrect.")
    else:
        return dataset
ticker = 'AAPL'
start_date = '2015-01-06' #YYYY-MM-DD
#end_date = datetime.date.today()
end_date = '2020-01-02'
dataset = get_data(ticker, start_date, end_date, 'stooq') #Main settings to change

#Cleaning Dataset
"""
    There will be missing weekdays due to holidays, here we will get those missing values
    by using the previous day closing prices.
    However, there is a problem, where if I take the dataset on a weekend, the first few values
    might appear as NaN.
"""
closing_prices = dataset['Close']
all_weekdays = pd.date_range(start = start_date, end = end_date, freq = 'B') #B is Business Days.
closing_prices = closing_prices.reindex(all_weekdays) #To include all missing dates.
closing_prices = closing_prices.fillna(method = 'ffill') #ffill stands for Forward Fill
closing_prices = closing_prices.to_frame() #Convert to Dataframe

#Plotting the Original Graph
# plt.plot(all_weekdays, closing_prices, color = 'blue')
# plt.tick_params(labelsize = '8') #Adjusting the xlabel and ylabel font size.
# plt.title("Line Chart for Closing Prices of " + ticker)
# plt.xlabel('Date')
# plt.ylabel("Closing Price ($)")
# plt.grid()
# plt.show()

#Shifting Values Out for Prediction and Comparison by using Past Prices to Reflect Future Prices
forecast_out = 20 #days
closing_prices['Prediction'] = closing_prices.shift(-forecast_out) #Shift down by n days

#X and y Values
X = closing_prices[:-forecast_out].drop(columns = 'Prediction')
y = closing_prices[:-forecast_out].drop(columns = 'Close')

#Splitting into Train/Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

#Benchmark Accuracy using Linear Regression
from sklearn.linear_model import LinearRegression
linear_r = LinearRegression()
linear_r.fit(x_train, y_train)
benchmark = linear_r.score(x_test, y_test)

#Using Support Vector Regression
from sklearn.svm import SVR
svr = SVR(kernel = 'linear')
svr.fit(x_train, np.ravel(y_train))
svr_accuracy = svr.score(x_test, np.ravel(y_test))

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rforest = RandomForestRegressor(random_state = 123)
rforest.fit(x_train, np.ravel(y_train))
rforest_accuracy = rforest.score(x_test, np.ravel(y_test))
rforest_train = rforest.predict(x_train)
rforest_test = rforest.predict(x_test)

#Cross Validation to Evaluate Models
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
    train_set = cross_val_predict(estimator = estimator, X = x_train, y = y_train, cv = 20)
    predict_result = cross_val_predict(estimator = estimator, X = x_test, y = y_test, cv = 20)
    predscore = metrics.r2_score(y_true = y_test, y_pred = predict_result)
    print(name + " Average Accuracy: " + str(round_to_2dp(accuracies.mean())) + "% (Train Set)")
    print(name + " Standard Deviation: " + str(round_to_2dp(accuracies.std())) + "% (Train Set)")
    print(name + " Average Accuracy: " + str(round_to_2dp(predscore)) + "% (Test Set)")
    return train_set, predict_result

linear_r_train, linear_r_test = cross_validation(linear_r, x_train, y_train, x_test, y_test, "Linear Regression")
svr_train, svr_test = cross_validation(svr, x_train, np.ravel(y_train), x_test, np.ravel(y_test), "SVR")
'Did not apply it on Random Forest as RF already performs averaging on its own.'

#Ensemble Models (Stacking and Boosting)
"""
Ensemble methods can decrease variance, bias or improve predictions.
There are three main types: Bagging, Boosting and Stacking.
Bagging: Averages *Random Forest is a type of Bagging.
Boosting: Covert weak learners to strong learners
Stacking: Combines (Best for this problem)

Within these types, there are two methods:
Sequential: Dependence between the base learners. Penalize mislabled examples.
Parallel: Independence between the base learners. Average Results
"""
def resize_array(array):
    array_size = array.size
    array = array.reshape(array_size, 1)
    return array
linear_r_train = resize_array(linear_r_train)
linear_r_test = resize_array(linear_r_test)
svr_train = resize_array(svr_train)
svr_test = resize_array(svr_test)
rforest_train = resize_array(rforest_train)
rforest_test = resize_array(rforest_test)

def dataframe_data(datasets):
    dataframe_data = np.concatenate(datasets)
    dataframe_data = np.ravel(dataframe_data)
    return dataframe_data
train_datasets = [linear_r_train, svr_train, rforest_train]
test_datasets = [linear_r_test, svr_test, rforest_test]
s_train = dataframe_data(train_datasets)
s_test = dataframe_data(test_datasets)

def duplicate_array(array):
    'estimator.fit requires s_train.size = xgb_train.size'
    array_train = array.append(array)
    array_train = array_train.append(array)
    return array_train
xgb_train = duplicate_array(y_train)
xgb_test = duplicate_array(y_test)

def create_dataframe(dataset, dataset_for_index, column_name):
    index = dataset_for_index.index.to_numpy()
    dataframe = pd.DataFrame(dataset, index = index, columns = [column_name])
    return dataframe
s_pdtrain = create_dataframe(s_train, xgb_train, 'Close')
s_pdtest = create_dataframe(s_test, xgb_test, 'Close')

estimator = xgb.XGBRegressor()
estimator.fit(s_pdtrain, xgb_train)
estimator_accuracy = estimator.score(s_pdtest, xgb_test)

#Predicting Future Prices by n days using Benchmark
future_weekdays = pd.date_range(start = end_date, periods = forecast_out + 1, freq = 'B')

def y_pred(estimator, prepared_dataset):
    """
    estimator = model
    returns future predicted prices.
    """
    forecast_out = 20
    forecast = prepared_dataset[-forecast_out:].drop(columns = 'Prediction')
    y_pred = estimator.predict(forecast)
    return y_pred
def plotting_data(y_pred, prepared_dataset, future_weekdays):
    """
    y_pred = future predicted prices
    returns data for plotting of charts (REQUIRES 2 CONTAINERS)
    """
    last_price = 1

    final_dataset = prepared_dataset.drop(columns = 'Prediction')
    last_price = final_dataset['Close'][-last_price:]
    y_pred = np.insert(y_pred, 0, last_price) #To connect the line between old and new prices
    future_prices = pd.DataFrame(data = {'Prediction': y_pred.flatten()}, index = future_weekdays)
    final_dataset = final_dataset.append(future_prices, sort = True)

    return final_dataset, future_prices

# linear_regression = y_pred(estimator = linear_r, prepared_dataset = closing_prices)
# linear_regression, lr_prices = plotting_data(linear_regression, closing_prices, future_weekdays)

# sp_vector_regression = y_pred(estimator = svr, prepared_dataset = closing_prices)
# sp_vector_regression, svr_prices = plotting_data(sp_vector_regression, closing_prices, future_weekdays)

prediction = y_pred(estimator, closing_prices)
prediction, prediction_prices = plotting_data(prediction, closing_prices, future_weekdays)

#Printing Models' Accuracy
print(
"Linear Regression Accuracy: " + str(round_to_2dp(benchmark)) + "%\n" +
"SVR Accuracy: " + str(round_to_2dp(svr_accuracy)) + "%\n" + 
"Random Forest Accuracy: " + str(round_to_2dp(rforest_accuracy)) + "%\n" +
"Model Accuracy: " + str(round_to_2dp(estimator_accuracy)) + "%"
)

#Plotting the Next n Days of Stock Price
def macroplot (name, all_weekdays, prepared_dataset, future_weekdays, future_prices, ticker):
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
    plt.suptitle(name, size = 12)
    plt.xlabel('Date')
    plt.ylabel("Closing Price ($)")
    plt.grid()
    plt.show()
# macroplot("Linear Regression", all_weekdays, closing_prices, future_weekdays, lr_prices, ticker)
# macroplot("Support Vector Regression", all_weekdays, closing_prices, future_weekdays, svr_prices, ticker)
macroplot("Final Model", all_weekdays, closing_prices, future_weekdays, prediction_prices, ticker)

#Plotting the Prediction Only
def microplot (name, future_weekdays, future_prices, ticker):
    """
    name = name of model
    future_weekdays = the weekdays that indexes the prices we have predicted.
    future_prices = the predicted prices
    ticker has been declared above.
    shows a line chart of only predicted prices and the the price the day before.
    """
    plt.plot(future_weekdays, future_prices, color = 'green') #Future Prices
    plt.tick_params(labelsize = '6') #Adjusting the xlabel and ylabel font size.
    plt.title("Line Chart for Predicted Prices of " + ticker)
    plt.suptitle(name, size = 12)
    plt.xlabel('Date')
    plt.ylabel("Closing Price ($)")
    plt.grid()
    plt.show()
# microplot("Linear Regression", future_weekdays, lr_prices, ticker)
# microplot("Support Vector Regression", future_weekdays, lr_prices, ticker)
microplot("Final Model", future_weekdays, prediction_prices, ticker)

#Export Data
export_data = prediction_prices.to_csv()
with open('Prediction.csv', 'w') as f:
    write_data = f.write(export_data)
    f.closed

#References
"""
    https://learndatasci.com/tutorials/python-finance-part-yahoo-finance-api-pandas-matplotlib/
    https://towardsdatascience.com/predicting-stock-prices-with-python-ec1d0c9bece1
    https://medium.com/@randerson112358/predict-stock-prices-using-python-machine-learning-53aa024da20a
    https://blog.statsbot.co/ensemble-learning-d1dcd548e936
    https://towardsdatascience.com/automate-stacking-in-python-fc3e7834772e
    https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python#Second-Level-Predictions-from-the-First-level-Output
"""