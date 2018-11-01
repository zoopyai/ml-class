# dbg1 - odsc class notes

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM, SimpleRNN, Dropout
from keras.callbacks import LambdaCallback

import wandb
from wandb.keras import WandbCallback

import plotutil
from plotutil import PlotCallback

wandb.init()
config = wandb.config

config.repeated_predictions = True
# dbg1 - this is only looking back 4 digit 
# (in text - it can be more.. this 4 may be less... this is how much lookback you need)
config.look_back = 4 

def load_data(data_type="airline"):
    if data_type == "flu":
        df = pd.read_csv('flusearches.csv')
        data = df.flu.astype('float32').values
    elif data_type == "airline":
        df = pd.read_csv('international-airline-passengers.csv')
        data = df.passengers.astype('float32').values
    elif data_type == "sin":
        df = pd.read_csv('sin.csv')
        data = df.sin.astype('float32').values
    return data

# convert an array of values into a dataset matrix
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-config.look_back-1):
        a = dataset[i:(i+config.look_back)]
        dataX.append(a)
        dataY.append(dataset[i + config.look_back])
    return np.array(dataX), np.array(dataY)

data = load_data("sin")
    
# normalize data to between 0 and 1
max_val = max(data)
min_val = min(data)
data=(data-min_val)/(max_val-min_val)

# split into train and test sets
split = int(len(data) * 0.70)
train = data[:split]
test = data[split:]

trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)

trainX = trainX[:, :, np.newaxis]
testX = testX[:, :, np.newaxis]

# create and fit the RNN
model = Sequential()
# dbg1 - this 1 (in as a first param of SimpleRNN is state vector size)
model.add(SimpleRNN(1, input_shape=(config.look_back,1 )))  # dbg1 - the last 1 is how many you are trying to predict.  If there are two deseases than that will be 2
# following is with 5 inputs (like in PDF)
model.add(Dense(1))
# dbg1 - model.add(SimpleRNN(5, input_shape=(config.look_back,1 )))  # dbg1 - the last 1 is how many you are trying to predict.  If there are two deseases than that will be 2
model.compile(loss='mae', optimizer='rmsprop')
model.fit(trainX, trainY, epochs=1000, batch_size=20, validation_data=(testX, testY),  callbacks=[WandbCallback(), PlotCallback(trainX, trainY, testX, testY, config.look_back)])





