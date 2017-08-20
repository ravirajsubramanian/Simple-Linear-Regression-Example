# Simple Regression Example
#
# Simple regression to assume the stock prices for next 30 days with existing past data.
#
# What do we use here,

# quandl - a open source library to get past data of stock prices of an organization,
#           no need for API key up to 50 requests/day
# sklearn (scikit-learn) - to use existing LinearRegression model and do some pre-progressing
# numpy - to build data in array format which is feasible to apply regression model
# matplotlib - to plot the graph for the data and assumption
# pickle - to save data and trained model and read back
# math, datetime, os - some standard python libraries
#

import math
import datetime
import os
import numpy as np
import pickle
import quandl
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')  # set graph style

df = None
filename = 'data.pickle'


def get_data():
    # before reading data make sure if the file exists
    if not os.path.exists(os.path.dirname('./' + filename)):
        save_data(df)  # fetch the data set and save when the file is not exist

    with open(filename, "rb") as file:
        return pickle.load(file)  # read the data set from file


def save_data(df):
    if df is None and not os.path.exists(os.path.dirname(filename)):
        df = quandl.get('WIKI/GOOGL')  # get Google data set from WIKI for example request
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # make sure if the path exists
        with open(filename, 'wb') as f:
            pickle.dump(df, f)  # write file with the data set


if df is None:
    df = get_data()  # get data set to train

if df is not None:
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]  # select columns to be used

    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0  # calculate High Low percentage

    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0  # change in percentage

    forecast_col = 'Adj. Close'  # column to be forecast
    df.fillna(-99999, inplace=True)

    forecast_out = int(math.ceil(0.01 * len(df)))  # set forecast length

    df['label'] = df[forecast_col].shift(-forecast_out)

    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)  # scale the data before taking as input
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    df.dropna(inplace=True)
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)  # init data

    clf = LinearRegression(n_jobs=-1)  # choose training model
    clf.fit(X_train, y_train)  # train

    trained_model = 'linear_regression.pickle'  # file to save trained model

    with open(trained_model, 'wb') as f:
        pickle.dump(clf, f)

    with open(trained_model, 'rb') as file:
        clf = pickle.load(file)  # get trained model

    accuracy = clf.score(X_test, y_test)  # calculate accuracy

    forecast_set = clf.predict(X_lately)  # assume forecast data with trained model
    print(forecast_set, accuracy, forecast_out)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400  # seconds in a day
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]  # join forecast data with past data

    # plot graph
    df['Adj. Close'].plot()
    df['Forecast'].plot()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
