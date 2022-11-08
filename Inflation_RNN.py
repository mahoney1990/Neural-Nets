# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:01:19 2022

@author: mahon
"""
import numpy as np
import pandas
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

def normalize(df,minmax=False):
    if minmax==False:
        normalized_df=(df-df.mean())/df.std()
    
    if minmax==True:
        normalized_df=(df-df.min())/(df.max()-df.min())
        
    return normalized_df


def gen_ts(data,frame=1):
    dataY=[]
    dataX=[]
    frame=1
    for i in range(len(data)-frame-1):      
        add=data[i:(i+frame),0]
        dataX.append(add)
        dataY.append(data[i+frame,0])
    
    return np.array(dataX), np.array(dataY)



#Import our time series
dataset = pandas.read_csv(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\Inflation_data.csv', engine='python')
N=len(dataset)

Y=dataset.pop('CPI').values
Y=Y.reshape(len(Y),1)
Y_lag, Y =gen_ts(Y)

X=dataset[['Govt','FFR','Stim1','Stim2','Stim3','War']].values
X=np.hstack(X,Y_lag)

X=normalize(X)
X=X.reshape(X.shape[0],X.shape[1],1)

print('X shape: '+str(X.shape))
print('Y shape: '+ str(Y.shape))

out_dim=1



#Assemble RNN
model=Sequential()
model.add(LSTM(24, activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(out_dim))
model.compile(loss="mse", optimizer="adam")

model.fit(X,Y,epochs=300,batch_size=5)

Y_hat=model.predict(X)

plt.plot(Y_hat)
plt.plot(Y)

np.sum((Y-Y_hat.flatten())**2)/N




