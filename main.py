# installs for the project
# pip install numpy pandas matplotlib pandas-datareader tensorflow scikit-learn keras

import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
import pandas_datareader as web

import datetime as dt


from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# from tensorflow.keras.layers import Dense, Dropout, LSTM
# from tensorflow.keras.models import Sequential

crypto_currency = 'BTC' # This is the mother of all crypto - most holders of other crypto cyrrencys have this as part of their "stuff"
official_currency = 'GBP'

start = dt.datetime(2015,1,1)

now = dt.datetime.now()

data = web.DataReader(f'{crypto_currency}-{official_currency}', 'yahoo', start , now)


# Pre P

print(data.head())

print(data)