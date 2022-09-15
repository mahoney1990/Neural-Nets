# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:50:12 2022

@author: mahon
"""
#Step one import data
#In this script we estimate cab-fare rates in NYC (Kaggle 2018 Competition)

import pandas as pd
import numpy as np

data=pd.read_csv('train.csv', parse_dates=['pickup_datetime'] ,nrows=500000 )

#Lets start by specifiying the borders of NYC
nyc_min_long=-74.05
nyc_max_long=-73.75

nyc_min_lat=40.63
nyc_max_lat=40.85

#Okay lets start by dropping observations that orignate or end outside NYC
df=data.copy(deep=True)
df=df[(df['pickup_longitude']> nyc_min_long) & (df['pickup_longitude']<nyc_max_long)]
df=df[(df['dropoff_longitude']> nyc_min_long) & (df['dropoff_longitude']<nyc_max_long)]

df=df[(df['pickup_latitude']> nyc_min_lat) & (df['pickup_latitude']<nyc_max_lat)]
df=df[(df['dropoff_latitude']> nyc_min_lat) & (df['dropoff_latitude']<nyc_max_lat)]


#Whoa! Negative fares. That can't be right. Lets drop em' (and reset our index!)
df=df[df.fare_amount>0]
df=df.reset_index()

#Now lets go ahead and plot all this shit.
import matplotlib.pyplot as plt

def texi_scatter(df, plot_type='pickup'):

    if plot_type=='pickup':
        plt.plot(list(df.pickup_longitude), list(df.pickup_latitude), '.', markersize=1) 
    else:
        plt.plot(list(df.dropoff_longitude), list(df.dropoff_latitude), '.', markersize=1) 

    plt.xlabel('Latitude')    
    plt.ylabel('Logitude')
    plt.show()

#Looks like New York....wild
#Define a distance fuction, and create a ride distance variable
#This is distance 'as the crow flies', but its still something

def dist(pickup_lat,pickup_long,dropoff_lat,dropoff_long):
    return np.sqrt(np.subtract(pickup_lat,dropoff_lat)**2+np.subtract(pickup_long,dropoff_long)**2)

df['ride_dist']=dist(list(df.pickup_latitude), list(df.pickup_longitude),
                     list(df.dropoff_latitude), list(df.dropoff_longitude))

df.ride_dist=2**(1/2)*df.ride_dist

#Okay now lets look at the 'datetime' variable varies

df['year']=df.pickup_datetime.dt.year
df['month']=df.pickup_datetime.dt.month
df['day']=df.pickup_datetime.dt.day

#We'll need to convert to day of the week in here somehow. Oh hell ya .dt coming through!
df['day_week']=df.pickup_datetime.dt.dayofweek

#And get time too
df['time']=df.pickup_datetime.dt.hour

#And a binary for holidays -- xmas and new years
df['holiday']=np.zeros(len(df))

for i in range(len(df)):
    if ((df.month[i]==12) & (df.day[i]==25)) or ((df.month[i]==12) & (df.day[i]==31)) or ((df.month[i]==1) & (df.day[i]==1)): 
        df.holiday[i]=1

#Now lets check for missing data and weird stuff
#This checks for missing values
print(df.isnull().sum())
df=df.dropna()

#Lets check our outcome variable range
print(min(df.fare_amount))
print(max(df.fare_amount))




#%%
#Okay now lets try visualizing our outcome some more
#Maybe cluster pickups that are close to an airport?


df.plot.scatter('fare_amount','ride_dist')

#I want to understand the clusters above and below the big upwardsloping mass...
#Lets dig into the fare rates

plt.hist(x=df.fare_amount, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)

#There are only 11 observations with rates above $100. Lets drop them. 
sum(df.fare_amount>100)
df=df[df.fare_amount<=100]

#What about passenger outliers?
min(df.passenger_count)
max(df.passenger_count)
sum(df.passenger_count==0)

#Ok so some of these are misrecorded? Replace with ones
df.passenger_count[df.passenger_count==0]=1

#Finally, we need to identify distance from airports. Airport taxis are more expensive
#Here are coordinates for three NYC airports. This ends up being pretty clutch
airports={'JFK': (-73.78,40.643), 'LaGuardia': (-73.87,40.77),'Newark': (-74.18,40.69)}

#And build two new distance variables
for i in airports:
    df['pickup_dist_'+i+'_aiport']=dist(list(df.pickup_latitude), list(df.pickup_longitude), 
                                        airports[i][1], airports[i][0])
    
    df['dropoff_dist_'+i+'_aiport']=dist(list(df.dropoff_latitude), list(df.dropoff_longitude),
                                         airports[i][1], airports[i][0])

#%%
####Part Two: Lets standardize our data so scales are not wildly off
import sklearn
import sklearn.model_selection
from sklearn import preprocessing

#Split out new, scaled data into testing and training sets
Y=df.loc[:,'fare_amount']

X=df.drop(['key','fare_amount','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']
          ,axis=1)

X=preprocessing.scale(X)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=.25)


#Lets build us a neural net boys

from keras.models import Sequential
from keras.layers import Dense

#Define a model object
model=Sequential()

#Add a couple layers
model.add(Dense(128, activation='relu',input_dim=X_train.shape[1]))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(8,activation='relu'))

#And the output layers
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

#And fit the model
model.fit(X_train,y_train, epochs=50)

#How did we do?
#Training
model.evaluate(X_train,y_train)

#Testing
model.evaluate(X_test,y_test)[1]

y_hat=model.predict(X_test).flatten()
MSE=sum(((y_test-y_hat)**2))/len(y_hat)
RMSE=MSE**(1/2)

print('Test RMSE: '+RMSE)










