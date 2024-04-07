import pandas as pd
import numpy as np
import datetime as dt
import vectorbt as vbt
import math
import talib
import requests

BASE_URL = 'https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09?apiKey=ppZ0Z1lnc4zbL6P1WQB0WVD5S5q0UGtV'

AGGREGATE_BARS_URL = 'https://api.polygon.io/v2/aggs/ticker/TSLA/range/1/hour/2023-12-01/2024-01-20?adjusted=true&sort=asc&limit=50000&apiKey=ppZ0Z1lnc4zbL6P1WQB0WVD5S5q0UGtV'

response = requests.get(f"{AGGREGATE_BARS_URL}")

data = response.json() #bars is a pything dictionary

bars = data['results']
print(len(bars))

df = pd.DataFrame()

for bar in bars:
    #we have to convert the unix milliseconds to seconds before we do anything
    time = (bar['t']) / 1000.0
    new_row = {'Time': dt.datetime.utcfromtimestamp(time), 'Close': bar['c']}
    df = df._append(new_row, ignore_index=True)

df.set_index('Time', inplace=True)
df.index = pd.to_datetime(df.index)


def get_smma(df, len, n):
    
    '''
    df:     dataframe
    len:    length of the timeperiod we want to analyze
    n:      length of the dataframe - 1
    '''
    
    #error case
    if n < len - 1:
        return 0
    
    #base case: we are at index corresponding to the length we are doing sma over 
    if n == len - 1:
        sliced_df = df['Close']
        base = talib.SMA(sliced_df, timeperiod = len).iloc[len-1]        
        return base
    
    else:
        #return smma sma(n-1) + close / n
        previous_smma = get_smma(df, len, n - 1)
        smma = ((previous_smma * (len-1)) + df['Close'][n]) / len
        return smma
    
# calculating zlema

def get_zlema(df, len, n):
    #n is the index (number) of the datapoint I want to analyze
    
    #error case:
    if (n < (len * 2) - 2):
        return 0
        
    # part of the dataframe I want to analyze
    spliced_df = df[df.index[0]: df.index[n]]   
             
    ema1 = talib.EMA(spliced_df['Close'], timeperiod = len)
    ema2 = talib.EMA(ema1, timeperiod = len)
    
    ema1 = ema1.iloc[-1]
    ema2 = ema2.iloc[-1]
    d = ema1 - ema2
    
    zlema = ema1 + d
    return zlema

smma_list = []
zlema_list = []
ma_length = 10

# smma_list = get_smma_efficient(df, 5)

for index, row in df.iterrows():
    parse = len(df[df.index < index])
    smma_list.append(get_smma(df, ma_length, parse))
    zlema_list.append(get_zlema(df, ma_length, parse))
    
df['SMMA'] = smma_list
df['MI'] = zlema_list
print(df)

md_list = []
for index, row in df.iterrows():
    md = 0
    # error case
    if len(df[df.index < index]) < 1:
        md = 0
        md_list.append(md)
        continue
    
    if row['SMMA'] == 0:
        md = 0
        md_list.append(md)
        continue
    
    if row['MI'] == 0:
        md = 0
        md_list.append(md)
        continue
    
    # getting the data I need
    spliced_df = df[df.index < index]
    mi = row['MI']
    
    # calculating hi & low:
    largest_closing_price = spliced_df['Close'].idxmax()
    hi = (spliced_df.loc[largest_closing_price]['SMMA'])

    lowest_closing_price = spliced_df['Close'].idxmin()
    lo = (spliced_df.loc[lowest_closing_price]['SMMA'])
    # print(f"hi: {hi} \nlow: {lo} \n\n")
    
    if mi > hi:
        md = mi - hi
        
    elif mi < lo:
        md = mi - lo
    
    else:
        md = 0
            
    md_list.append(md)

df['MD'] = md_list

#sb: calculate the simple moving average of md and the signal length
sb_list = []
signal_length = 5

for index, row in df.iterrows():
    
    sb = 0
    
    #error case: 
    if len(df[df.index < index]) < signal_length:
        sb = 0
        sb_list.append(sb)
        continue
    
    spliced_df = df[df.index < index]
    sb = talib.SMA(spliced_df['MD'], signal_length)
    sb_list.append(sb[-1])

df['Signal'] = sb_list

# Calculating the when the signal line and md cross over
# I need to know this to buy and sell
difference_list = []

for index, row in df.iterrows():
    difference = row['MD'] - row['Signal']
    difference_list.append(difference)

df['Difference'] = difference_list
 
action_list = []   
previous_index = None

for index, row in df.iterrows():
    #I want to check if the difference column changes signs
    #error case
    if (df.index[0] == index):
        action_list.append(0)
        previous_index = index
        continue
    else:
        previous_dfference = df.loc[previous_index]['Difference']
        previous_dfference = math.copysign(1, previous_dfference)
        
        current_difference = row['Difference']
        current_difference = math.copysign(1, current_difference)
        
        if previous_dfference < current_difference: 
            action_list.append(-1)
        elif previous_dfference > current_difference:
            action_list.append(1)
        else:
            action_list.append(0)
        
        previous_index = index

df['Action'] = action_list
  
csv_file_path = 'output.csv'
df.to_csv(csv_file_path, index=False)  # Use index=False to exclude row numbers as an additional column

def impulse_ma_strategy(Action):
    signal = Action
    return signal

my_indicator = vbt.IndicatorFactory(
    class_name="impulse_ma_strategy",
    short_name="impulse_ma",
    input_names=["Action"],
    output_names=["signal"]
    ).from_apply_func(
        impulse_ma_strategy
        )

results = my_indicator.run(df['Action'])
entries = results.signal == 1.0
exits = results.signal == -1.0
print(results)

pf = vbt.Portfolio.from_signals(df['Close'], entries, exits)
print()
print(pf.stats())
print(pf.plot().show())