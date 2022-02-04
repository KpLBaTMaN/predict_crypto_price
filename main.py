# installs for the project
# pip install numpy pandas matplotlib pandas-datareader tensorflow scikit-learn keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web

import datetime as dt
import random
import os

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


# Preprocessing
print(data.head())
print(data)

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

print(scaled_data)




def preprocess_data(data,scaler):
    
    # Strip the prices from the dataset
    prices = data['Close'].values
    
    # Preprocessing
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    return scaled_data
    #print(len(scaled_data))
    #print(type(scaled_data))
    
    
    
def process_data(data, training_days, future_days):
    
#     training_days = 60
#     future_days = 30

    x_train, y_train = [], []

    for x in range(training_days, len(data)-future_days):
        x_train.append(data[x-training_days:x,0]) #between date and before the prediction days
        y_train.append(data[x+future_days,0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
    
    return x_train, y_train


def Create_Neural_Network(x_train):
    model = Sequential()

    model.add(LSTM(units=50, return_sequences= True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences= True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    #print(len(model.layers))
    print(model.summary())
    
    return model


# #This will take in 3 variables 
# - crypto_currency = 'BTC' # This is the mother of all crypto - most holders of other crypto cyrrencys have this as part of their "stuff"
# - official_currency = 'GBP'
# - Starting Date - date the data will begin at.
def instance_crypto(crypto_currency,official_currency,start, training_days ,future_days):
    
    now = dt.datetime.now()
    
    
    ###### Obtain the data
    dataset = web.DataReader(f'{crypto_currency}-{official_currency}', 'yahoo', start , now)
    #print(dataset.head())
    
    
    #####Preprocesssing
    scaler = MinMaxScaler(feature_range=(0,1))
    p_dataset = preprocess_data(dataset,scaler)
    
    
    ######Process into training data
    x_train, y_train = process_data(p_dataset,training_days, future_days)
    
   
    ######Create Model
    model = Create_Neural_Network(x_train)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=100, batch_size=32)
    
    ######Test 
    test_start = dt.datetime(2020,1,1)
    test_end = dt.datetime.now()

    test_data = web.DataReader(f'{crypto_currency}-{official_currency}', 'yahoo', test_start , test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((dataset['Close'], test_data['Close']), axis=0)


    model_inputs = total_dataset[len(total_dataset) - len(test_data) - training_days:].values

    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = []

    for x in range(training_days, len(model_inputs)):
        x_test.append(model_inputs[x-training_days:x,0])
        
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    

    ####Prediction
    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    #####Display Data
    plot_figure(crypto_currency, official_currency, actual_prices, prediction_prices, future_days)
    
    
def plot_figure(crypto_currency, official_currency, actual_prices, prediction_prices, future_days):
    plt.figure()
    plt.plot(actual_prices, color = 'black', label = 'Actual Prices')
    plt.plot(prediction_prices, color='green', label='Predictied Prices')
    plt.title(f'{crypto_currency} price prediction ({official_currency})')
    plt.xlabel('Time (per day)')
    plt.ylabel(f'Price ({official_currency}) ')
    plt.legend(loc='upper left')
    
    figure = plt.gcf() # get current figure
    figure.set_size_inches(8, 6)
    
    
    directory = f'./{crypto_currency}'
    
    if os.path.exists(directory):
        #Export plot as png file
        plt.savefig(f'{directory}/{crypto_currency}_{official_currency}_{future_days}days.png', dpi = 100)
    else:
        os.mkdir(directory)
        #Export plot as png file
        plt.savefig(f'{directory}/{crypto_currency}_{official_currency}_{future_days}days.png', dpi = 100)

    
    plt.show()
    
    
    
def main():
#     crypto_currency = 'BTC' # This is the mother of all crypto - most holders of other crypto cyrrencys have this as part of their "stuff"
#     official_currency = 'GBP'
#
    start = dt.datetime(2015,1,1)
    training_days = 60
    future_days = 0
    
    listOfDays = [0,1,2,3,4,5,6,7,14,30]
    
    for days in listOfDays:
        instance_crypto('BTC', 'GBP', start, training_days, days)
        instance_crypto('ETH', 'GBP', start, training_days, days)
        instance_crypto('MATIC', 'GBP', start, training_days, days)
    
    

main()