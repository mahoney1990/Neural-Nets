# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:18:19 2022

@author: mahon
"""

import pandas
import matplotlib.pyplot as plt

#Univariate time series data
dataset = pandas.read_csv(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\SP500.csv', usecols=[1], engine='python')
dataset=dataset.values

#Remove no trading days
idx=dataset!='.'
dataset=dataset[idx]

#Reshape and convert to float32 (loaded in as object type)
dataset=dataset.reshape(len(dataset),1)
dataset=dataset.astype('float32')

#Plot a basic TS
plt.plot(dataset)
plt.show()

#We're gonna need some packages here
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#Alright we want to predict the SP500 index at t given all data before t
#Lets normalize the dataset real fast

max(dataset)
min(dataset)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


N=len(dataset)

train_size=int(.66*N)
test_size=N-train_size

train=dataset[0:train_size]
test=dataset[train_size:N]


import numpy as np

frame=1


#Careful here, your data needs the right shape
def gen_ts(data,frame=1):
    dataY=[]
    dataX=[]
    frame=1

    for i in range(len(data)-frame-1):      
        add=data[i:(i+frame),0]
        dataX.append(add)
        dataY.append(data[i+frame,0])
    
    return np.array(dataX), np.array(dataY)



#Generate TS for test and training
trainX, trainY = gen_ts(train)
testX, testY =gen_ts(test)


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#Lets build us a neural net!
model=Sequential()
model.add(LSTM(4,input_shape=(1,frame)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=10, verbose=2)

#Lets test it
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#Looks really good!
plt.plot(test)
plt.plot(testPredict)
plt.show

plt.plot(train)
plt.plot(trainPredict)
plt.show



