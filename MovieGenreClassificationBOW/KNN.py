import glob
import collections
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from clarifai.rest import Video as ClVideo
from math import*
from decimal import Decimal
import scipy.spatial.distance as sp
from sklearn import svm, preprocessing


def train(X_train):
    app = ClarifaiApp(api_key = 'e9dd831462924e7f94a7d4bed506cade')

    imgs = sorted(glob.glob('D:/python/video frame/romance-tr/Cup of Love - Official Trailer - MarVista Entertainment (3-25-2018 9-22-09 PM)/*.jpg'))
    
    arr = np.zeros(3000)
    bag_of_words = []
    X_train = np.loadtxt('training data.txt',dtype = int)

    #load bag of words after running 1 film
    file = open('bag_of_words.txt','r')
    for j in file:
        if(j != '\n'):
            bag_of_words.append(j.strip('\n'))
    file.close()

    #processing video frames
    for img in imgs:
        tags = app.models.get("general-v1.3").predict_by_filename(img)
        for tag in tags['outputs'][0]['data']['concepts']:
            if tag['name'] in bag_of_words:
                arr[bag_of_words.index(tag['name'])] += 1
            else:
                bag_of_words.append(tag['name'])
                arr[bag_of_words.index(tag['name'])] += 1
                f = open('bag_of_words.txt','a')
                f.write(tag['name']+'\n')
                f.close()
    s = np.sum(arr)
    if(s != 0):
        f1 = open('label_train.txt','a')
        f1.write('romance' + '\n')
        f1.close()

        X_train = np.vstack([X_train,arr])
        np.savetxt('training data.txt',X_train)
        print("successful!")
    else:
        print("error!")

#-----------------------------------KNN-----------------------------------------#
def predict(X_train, y_train, x_test, k, n):
        # create list for distances and targets
        distances = []
        label = []
        top = []
        for i in range(len(X_train)):
                # compute the distance
                if(n == 0):
                    distance = sp.minkowski(x_test, X_train[i, :], p = 1)#np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
                elif(n == 1):
                    distance = sp.euclidean(x_test, X_train[i, :])
                else:
                    distance = sp.cosine(x_test, X_train[i, :])
                # add it to list of distances
                distances.append([distance, i])

        # sort the list
        distances = sorted(distances)
        
        # make a list of the k neighbors' targets
        check = 1
        for i in range(k):
                index = distances[i][1]
                label.append(y_train[index])
                if(i > 1 and (label[-1] != label[-2])):
                        check = 0
                top.append(X_train[index])

        '''if(check == 0):
                x_test = np.reshape(x_test, (-1,2000))
                clf = svm.SVC(decision_function_shape='ovo', gamma = 0.0005)
                #clf = svm.LinearSVC()
                clf.fit(top, label)
                return clf.predict(x_test)[0]
        else:'''
        return collections.Counter(label).most_common(1)[0][0]
        

sum = 0
resultK = []
resultA = []
def accuracy(predictions, label_test, k):
        global sum
        global resultK, resultA
        cnt = 0
        for i in range(len(predictions)):
                if(predictions[i] == label_test[i]):
                        cnt+=1
        s = cnt/len(predictions)
        sum+=s
        resultK.append(k)
        resultA.append(s)
        #print('accuracy with k =',k,': ',s,'%')

def kNearestNeighbor(X_train, y_train, X_test, y_test, k, n):
        # train on the input data => file test1
        predictions = []
        
		# loop over all observations
        for i in range(len(X_test)):
                predictions.append(predict(X_train, y_train, X_test[i, :], k, n))
                #print(predictions[i] + '\t' + y_test[i])
                
        accuracy(predictions, y_test, k)

#-----------------------------------------loaddata-------------------------------------------#

X_train = np.array([],dtype = int)
X_test = np.array([],dtype = int)
y_train = []
y_test = []
t = 0

X_train = np.loadtxt('training data.txt', dtype = int)
#X_train = np.sqrt(X_train)
#X_train = preprocessing.normalize(X_train, norm = 'l1')
X_test = np.loadtxt('test data.txt', dtype = int)
#X_test = np.sqrt(X_test)
#X_test = preprocessing.normalize(X_test, norm = 'l1')
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

#------------------------------execution----------------------------#
#train(X_train)
'''start_time = time.time()
kNearestNeighbor(X_train, y_train, X_test, y_test, 10, 0)
t = time.time() - start_time'''
for i in range(1,49):
	start_time = time.time()
	kNearestNeighbor(X_train, y_train, X_test, y_test, i, 0)
	t += time.time() - start_time
plt.plot(resultK, resultA, color = 'r', linewidth = 1.0, label = 'minkowski( p = 1)')
resultK.clear(), resultA.clear()
for i in range(1,49):
        kNearestNeighbor(X_train, y_train, X_test, y_test, i, 1)
plt.plot(resultK, resultA, color = 'b', linewidth = 1.0, label = 'euclidean')
resultK.clear(), resultA.clear()
for i in range(1,49):
        kNearestNeighbor(X_train, y_train, X_test, y_test, i, 2)
plt.plot(resultK, resultA, color = 'g', linewidth = 1.0, label = 'cosine')
plt.legend()
plt.ylabel('accuracy (No SVM')
plt.xlabel('Value of K')
plt.show()
print('running time: ', t/48)
