import glob
import os
import collections
import subprocess
import numpy as np
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from clarifai.rest import Video as ClVideo
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
import time

def accuracy(predictions, label_test):
        cnt = 0
        for i in range(len(predictions)):
                #print(predictions[i] + '\t' + label_test[i])
                if(predictions[i] == label_test[i]):
                        cnt+=1
        print('accuracy: ',cnt*100.00/len(predictions),'%')

def Multinomial_NaiveBayes(X_train, y_train, X_test, y_test):
    result = []
    
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True) # laplace smmothing

    for i in range(len(X_test)):
        result.append(str(clf.predict(X_test)[i]))

    accuracy(result, y_test)


X_train = np.array([],dtype = int)
X_test = np.array([],dtype = int)
y_train = []
y_test = []

X_train = np.loadtxt('training data.txt', dtype = int)
X_test = np.loadtxt('test data.txt', dtype = int)

f0 = open('label_train.txt', 'r')
for i in f0:
	if(i != '\n'):
		y_train.append(i.strip('\n'))
f0.close()

f1 = open('label_test.txt', 'r')
for i in f1:
	if(i != '\n'):
		y_test.append(i.strip('\n'))
f1.close()

X_train = np.sqrt(X_train)
X_test = np.sqrt(X_test)
#X_train = preprocessing.normalize(X_train, norm = 'l1')
#X_test = preprocessing.normalize(X_test, norm = 'l1')
start_time = time.time()
Multinomial_NaiveBayes(X_train, y_train, X_test, y_test)
print('running time: ', time.time() - start_time)
        


