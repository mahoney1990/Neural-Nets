# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:14:20 2022

@author: mahon
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:18:19 2022

@author: mahon
"""

import pandas
import matplotlib.pyplot as plt
import numpy as np

#Univariate time series data
dataset = pandas.read_csv(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\Inflation_data.csv', usecols=range(1,16), engine='python')
col_names=list(dataset.columns)
dataset=dataset.values

#Pull corresponding dates
dates = pandas.read_csv(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\Inflation_data.csv', usecols=[0], engine='python')

#Reshape and convert to float32 (loaded in as object type)
dataset=dataset.astype('float32')


#We're gonna need some packages here
# L O A D I N G
##################################################
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
###################################################

#Alright we want to predict the CPI inflation at t given all data before t
#Get number of months real quick
N=len(dataset)

#Split data into train and test -- this is time series data so you can't do a random split
CPI=dataset[:, 0]
CPI=CPI.reshape(N,1)

#Clean up dates, convert to usable list
import datetime as dt
dates=np.array(dates)
dates=dates[:127]
dates=dates.flatten()

dates_list = [dt.datetime.strptime(date, "%m/%d/%Y").date() for date in dates]

#Define Lag variable(s)
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


#Generate TS for CPI
trainX, trainY = gen_ts(CPI)

dateset=dataset[0:N-frame-1,:]

#Now, lets weld in some other predicitve data
dataset=np.delete(dataset,0,1)
trainX = np.hstack((trainX, dataset[0:(N-frame-1),:]))

#Lets normalize the dataset real fast
scaler = MinMaxScaler(feature_range=(0, 1))
trainX = scaler.fit_transform(trainX)

#Lets drop the AR piece and see how we do

trainX=np.delete(trainX,0,1)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

N=trainX.shape[0]
time_dim=trainX.shape[1]
feature_dim=trainX.shape[2]


#Lets build us a neural net!
model=Sequential()
model.add(LSTM(24, input_shape=(1,feature_dim)))
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam',run_eagerly=())
model.fit(trainX, trainY, epochs=5000, batch_size=20, verbose=2)

#Lets test it
trainPredict = model.predict(trainX)

model.predict(trainX[0:2,:])

#Looks pretty good!
import matplotlib.dates as mdates

plt.plot(dates_list,CPI[1:128])
plt.plot(dates_list,trainPredict)
plt.gcf().autofmt_xdate()
plt.show()

import shap
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

DE=shap.DeepExplainer(model, trainX)
shap_values = DE.shap_values(trainX) # X_validate is 3d numpy.ndarray

shap_values=np.array(shap_values).reshape(N,feature_dim)

shap.initjs()
shap.summary_plot(
    shap_values, 
    trainX.reshape(N,feature_dim),
    feature_names=col_names[1:],
    max_display=10,
    plot_type='bar')

print('August Predicted Inflation:')
model.predict(trainX[126].reshape(1,1,feature_dim))



import  statsmodels
statsmodels.hpfilter(dataset['PRC_Imports'],100)



