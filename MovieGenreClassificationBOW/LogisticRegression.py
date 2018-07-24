from __future__ import division, print_function, unicode_literals
import numpy as np
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from clarifai.rest import Video as ClVideo
import time
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------------#
eta = 0.05
X = np.loadtxt("training data.txt", dtype=int)
x = np.ones((X.shape[0],1))
X_train = np.c_[x,X]                     #extend data
y_train = []
w_init = np.random.randn(X_train.shape[1], 1)

file = open("label_train.txt",'r')
for i in file:
    if(i != '\n'):
        y_train.append(i.strip('\n'))
file.close()

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def logisticRegression(X_train, y_train, w_init, eta, tol = 1e-4, max = 10000):
    N = X_train.shape[0]
    w = w_init
    cnt = 0
    while cnt < max:
        id = np.random.permutation(N)
        for i in id:
            xi = X_train[i].reshape(X_train.shape[1],1)
            yi = 0
            if(y_train[i] == 'horror'):
                yi = 1
            zi = sigmoid(np.dot(w.T,xi))
            w_new = w + eta*(yi - zi)*xi
            cnt+=1
            if(np.linalg.norm(w_new - w) < tol):
                print(cnt)
                return w
            w = w_new.copy()
    print(cnt)
    return w

def accuracy(predictions, label_test):
        cnt = 0
        for i in range(len(predictions)):
                #print(predictions[i] + '\t' + label_test[i])
                if(predictions[i] == label_test[i]):
                        cnt+=1
        print('accuracy: ',cnt*100.00/len(predictions),'%')

def classify(X_test, y_test):
    X_test = np.loadtxt("test data.txt", dtype=int)
    x = np.ones((X_test.shape[0],1))
    X_test = np.c_[x,X_test]   
    file = open("label_test.txt", 'r')
    for i in file:
        if(i != '\n'):
            y_test.append(i.strip('\n'))
    file.close()
    predict = []
    w1 = np.loadtxt("logistic1.txt")
    w2 = np.loadtxt("logistic2.txt")
    w3 = np.loadtxt("logistic3.txt")
    for i in range(len(X_test)):
        t1 = np.dot(w1,X_test[i])
        t2 = np.dot(w2,X_test[i])
        t3 = np.dot(w3,X_test[i])
        t = max(t1,t2,t3)
        if(t == t1):
            predict.append("action")
        elif(t == t2):
            predict.append("comedy")
        else:
            predict.append("horror")
    
    accuracy(predict,y_test)
        
        
w = logisticRegression(X_train, y_train, w_init, eta)
print(w)
X_test = np.array([],dtype = int)
y_test = []
classify(X_test, y_test)

       
        
            
        
    
    
    
    
    
    
