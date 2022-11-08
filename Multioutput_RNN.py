# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:17:46 2022

@author: mahon
"""

from numpy.random import uniform
import matplotlib.pyplot as plt
import numpy as np
from numpy import hstack

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

def normalize(df,minmax=False):
    if minmax==False:
        normalized_df=(df-df.mean())/df.std()
    
    if minmax==True:
        normalized_df=(df-df.min())/(df.max()-df.min())
        
    return normalized_df

N=100

#Initlaize input/output vectors
x1=[] 
x2=[] 
x3=[]
y1=[]
y2=[]

#Generate some data
for i in range(N):
    x1.append(i/100+uniform(-.1,.1))
    x2.append(i/100+uniform(-3,5)+2)
    x3.append(uniform(0,1)/np.sqrt(i+1))
    
    y1.append(2*x1[i]-.5*x2[i]+x3[i]+uniform(-1,1))
    y2.append(x1[i]+3*x3[i]+5+uniform(-1,3))

#Convert lists to numpy arrays
x1=np.array(x1).reshape(N,1)
x2=np.array(x2).reshape(N,1)
x3=np.array(x3).reshape(N,1)

y1=np.array(y1).reshape(N,1)
y2=np.array(y2).reshape(N,1)

#Assemble into matrices
X = hstack((x1, x2, x3))
Y = hstack((y1, y2))

X=normalize(X)

print(X.shape)
print(Y.shape)


X=X.reshape(X.shape[0],X.shape[1],1)

in_dim = (X.shape[1], X.shape[2])
out_dim = Y.shape[1]

#LSTM requires a 3 dimensional array, so lets reshape everything

model=Sequential()
model.add(LSTM(64,activation='relu'))
model.add(Dense(out_dim))

model.compile(loss="mse", optimizer="adam")

model.fit(X,Y, epochs=100, batch_size=12, verbose=1)

y1_hat=model.predict(X)[:,0]
y2_hat=model.predict(X)[:,1]

plt.plot(y1)
plt.plot(y1_hat)


