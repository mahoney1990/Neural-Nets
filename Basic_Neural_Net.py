# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 12:54:58 2022

@author: mahon
"""

###Lets build our own neural net from scratch (it'll be fun they said....)

import numpy as np

#Define everyone's favorite activation function + derivative
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoid_deriv(x):
    return x*(1-x)

#Now lets define our Neural Net as a class
class NeuralNet:
    
    #x and y are input and target values. We itizlatize weights randomly
    def __init__(self, x ,y ):
        self.input = x
        self.weight1= np.random.rand(self.input.shape[1],4)
        self.weight2=np.random.rand(4,1)
        self.y=y
        self.output=np.zeros(self.y.shape)

    #Define the feedforward components 
    def feed_forward(self):
        self.layer1= sigmoid(self.input @ self.weight1)
        self.output=sigmoid(self.layer1 @ self.weight2) 
        
    #Now we need to calculate graidents for our back propogation components 
    def back_prop(self):
        #We use the chain rule here to get our loss function in terms of weights
        #Then we just update weights by moving in the direction of greatest loss reduction. Easy right?
        
        #MAAAAAATTTTTHHH!!
        diff_weights2 = np.dot(self.layer1.T ,(2*(self.y-self.output)
                        *sigmoid_deriv(self.output)))
        
        
        diff_weights1=  np.dot(self.input.T, (np.dot(2*(self.y-self.output)
                        *sigmoid_deriv(self.output),self.weight2.T)
                        *sigmoid_deriv(self.layer1)))
       
        self.weight1+=diff_weights1
        self.weight2+=diff_weights2
        
#Okay that was a little tedious. Lets see if it works...

#%% Stat a new section cuz...why not
__name__="main"

if __name__ == "main":
    
    X=np.array([[0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]])

    y=np.array([[0],[1],[1],[0]])

    net=NeuralNet(X, y)

    for i in range(1500):
         net.feed_forward()
         net.back_prop()
    
    print(net.output)
        
#Looks pretty good!      
        
        
        
        
        
        