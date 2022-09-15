# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:39:12 2022

@author: mahon
"""

##Practice project #1##
#Predicting diabetes incidence

import pandas as pd
import numpy as np
data=pd.read_csv("diabetes.csv")

print(data.head)

#As always lets dig into these data

from matplotlib import pyplot as plt

#Histogram plots first. Couple things....
# 1. Looks like everything is bounded below by zero
# 2. Most variables look like a decreasing exponential, but some are normal
# 3. Everything is on a different scale -- we'll have to address that

data.hist()
plt.show()

#...Not sure what skin thinkness has to do with diabetes, but alright.
#Next lets form some kerenl density plots by diabeteic/nondiabetic
import seaborn as sns

plt.subplots(3,3,figsize=(15,15))

for idx, column in enumerate(data.columns):    
    sns.kdeplot(data=data, x=column, hue="Outcome")
    plt.show()

#Certainly glucose looks like its differs in mean. Pregnancy and BMI might too. I'm guessing
#these will get extra weight in the net.


#We'll want to check for missing values next
print(data.isnull().any()) #Looks like were good

#What about zero values? That could be a problem.
for col in data.columns:
    missing_rows= data.loc[data[col]==0].shape[0] #This piece tell us how many zero rows were have for each variable
    print(col + " -- " +str(missing_rows))

#Replace with means...for now. Well damn insulin shouldn't equal zero. Nor should some others
cols=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

for col in cols:
    data[col]=data[col].replace(0,np.nan)
    
#Ok now lets replace with means
for col in cols:
    data[col]= data[col].fillna(data[col].mean()) 

#%%
####Part Two: Lets standardize our data so scales are not wildly off
import sklearn
from sklearn import preprocessing
data_scaled=preprocessing.scale(data)
data_scaled=pd.DataFrame(data_scaled, columns= data.columns)

#Remember to turn your binary features back to 0,1!
data_scaled['Outcome']=data['Outcome']

#Split out new, scaled data into testing and training sets
X=data_scaled.loc[:, data_scaled.columns !='Outcome']
Y=data_scaled.loc[:,'Outcome']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=.25)

#Lets build us a neural net boys

from keras.models import Sequential
from keras.layers import Dense

#Define a model object
model=Sequential()

#Add a couple layers
model.add(Dense(32, activation='relu',input_dim=8))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
#And the output layers
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#And fit the model
model.fit(X_train,y_train, epochs=250)

#How did we do?
#Training
model.evaluate(X_train,y_train)

#Testing
model.evaluate(X_test,y_test)[1]

#%% Post-hoc analysis
#Check out our classification errors. Lets build a heat-map confusion matrix

from sklearn.metrics import confusion_matrix

#Generate binary predictions
y_hat=(model.predict(X_test) > 0.5).astype("int32")
c_mat=confusion_matrix(y_test, y_hat)

mat=sns.heatmap(c_mat, annot=True,
                xticklabels=['No Diabetes','Dibetes'],
                yticklabels=['No Diabetes','Dibetes'],
                cbar=False,
                cmap='Blues')

mat.set_xlabel('Prediction')
mat.set_ylabel('Actual')

#ROC Curve

vals=[.1,.2,.3,.4,.5,.6,.7,.8,.9,.99]
TPR=np.zeros(len(vals))
FPR=np.zeros(len(vals))

for i in range(len(vals)):
    y_hat=(model.predict(X_test) > vals[i]).astype("int32")
    
    c_mat=confusion_matrix(y_test, y_hat)
    
    TPR[i]=c_mat[1,1]/sum(c_mat[1,:])
    FPR[i]=c_mat[0,1]/sum(c_mat[0,:])



