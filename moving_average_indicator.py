import numpy as np
import pandas as pd
import datetime as dt
import requests
import talib
import math
import yfinance as yf

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import QuantileTransformer


# Classes

# First, we are going to get data from our Finance API (Polygon)
# We are going to get 'aggregate bars'
# This is from AAPL

api_key = ''
polygon_data = api_key

response = requests.get(f"{polygon_data}")
polygon_data = response.json()

result_bars = polygon_data['results'] # this is a dictionary

aggregate_bars = pd.DataFrame()

# we are turning this dictionary into a pnadas dataframe to make it easier to use 
for bar in result_bars:
    #we have to convert the unix milliseconds to seconds before we do anything
    time = (bar['t']) / 1000.0
    new_row = {'Date': dt.datetime.utcfromtimestamp(time), 'Close': bar['c']}
    aggregate_bars = aggregate_bars._append(new_row, ignore_index=True)

aggregate_bars.set_index('Date', inplace=True)
aggregate_bars.index = pd.to_datetime(aggregate_bars.index)
aggregate_bars.index = aggregate_bars.index.date #this is taking away the time component

# Second, we are calculating the moving average and recording the action 
# this is not giving me a correct moving average -> why? 
aggregate_bars['MACD'], aggregate_bars['MACD_Signal'], macdhist = talib.MACDFIX(aggregate_bars['Close'], signalperiod=9)

qt = QuantileTransformer(output_distribution='normal', random_state=0)
aggregate_bars["Difference"] = aggregate_bars['MACD'] - aggregate_bars['MACD_Signal']

# I want to compile a list of times I want to buy and I want to sell, 
# and I am going to add it to this pandas dataframe 
# then I'm going to feed this pandas dataframe into a machine learning model, 
# and try to train the model how to buy and sell based moving averages 

# Here is how I want to format my data
# September 01 - November 30 is my Training Data
# December 01 - January 31 is my Testing Data
# Record everywhere on that chart I would buy and sell
# Train a model with your MA parameters as the input and the 'buy/sell' action as the output
# Test the model against the January Data
# NOTE: this is going to be a month long trading strategy
# if I want a high frequency trading alogithm, I'm going to have to lower the time interval by A LOT

#Third, adding the dates where I want to buy & sell

action_list = [0] * aggregate_bars.shape[0]
aggregate_bars['Action'] = action_list

#Note to self: you should probably store these numbers in a text file and just find a way to import them
buy_list = ['23-01-5', '23-01-6',
            '23-3-1', '23-3-02', 
            '23-3-13', '23-3-14',
            '23-4-12', '23-4-13',
            '23-4-25', '23-4-26',
            '23-5-3', '23-5-4',
            '23-5-23', '23-5-24',
            '23-6-7', '23-6-8',
            '23-6-21', '23-6-22',
            '23-7-11', '23-7-12',
            '23-8-17', '23-8-18',
            '23-9-13', '23-9-14',
            '23-9-27', '23-9-28',
            '23-10-26', '23-10-27',
            '23-12-4', '23-12-5',
            '24-1-4', '24-1-5',
            '24-1-16', '24-1-17']

sell_list = ['23-2-7','23-2-8',
             '23-3-6', '23-3-7',
             '23-4-3', '23-4-4',
             '23-4-18', '23-4-19',
             '23-5-1', '23-5-2',
             '23-4-18', '23-4-19',
             '23-6-5', '23-6-6',
             '23-6-15', '23-6-16',
             '23-7-3',
             '23-7-20', '23-7-21',
             '23-9-5', '23-9-6',
             '23-9-19', '23-9-20',
             '23-10-12', '23-10-13',
             '23-11-20', '23-11-21',
             '23-12-13', '23-12-14',
             '24-1-10', '24-1-11',
             '24-1-23', '24-1-24']

buy_list = [dt.datetime.strptime(date, '%y-%m-%d') for date in buy_list]
sell_list = [dt.datetime.strptime(date, '%y-%m-%d') for date in sell_list]
aggregate_bars.loc[buy_list, 'Action'] = 1
aggregate_bars.loc[sell_list, 'Action'] = -1

# Creating Training & Testing Data
aggregate_bars = aggregate_bars.dropna()
aggregate_bars = aggregate_bars.loc[aggregate_bars['Difference'] < 0.5]
aggregate_bars = aggregate_bars.loc[-0.5 < aggregate_bars['Difference']]


train = aggregate_bars.loc[aggregate_bars.index <= dt.datetime.strptime('12-30-23', '%m-%d-%y').date()]
test = aggregate_bars.loc[aggregate_bars.index > dt.datetime.strptime('12-30-23', '%m-%d-%y').date()]

features = ['Difference']
target = ['Action']

X_train = train[features]
y_train = train[target]
y_train = y_train.values.ravel()

X_test = test[features]
y_test = test[target]
y_test = y_test.values.ravel()

# Training our Model

k_values = range(1, 15)
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=2, scoring='accuracy').mean()
    scores.append(score)

best_k = k_values[np.argmax(scores)]
print(f"Best k value: {best_k}")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Predict on the test set

y_pred = knn.predict(X_test)
print(y_pred)

# Evaluate the model

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Adding it to our dataframe

aggregate_bars['Prediction'] = knn.predict(aggregate_bars[features])
aggregate_bars.to_csv('ma_indicator.csv')

'''
WHAT HAVE WE LEARNED

when combining KNN with the MA parameter, we don't get a lot of good predictions

KNN does not work well when there is a lot of dimension of data
AND
when the data is very different in scale

you need to scale the data when it comes to machine learning!
get comfortable using the standard scaler 

Note to self: when scaling, its good practice to NOT scale your target data

There are also A LOT OF zero's for the holding value. It is affecting how the model determines whether or not to buy/sell

The difference between each 'difference' value is small. You need to transform the data to spread it out

I need A LOT more datapoints that correspond to a buy or sell signal

I need to find a way to combine all the 'Action'=0 datapoints

I also need a better parameter -> moving average really doesn't tell me that much!

Future Implimentations: 
Save this data to a SQL database & create an API that can access this database
'''