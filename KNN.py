## Importing Libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import accuracy_score

#Load breascancer dataset from sklearn datasets
data=load_breast_cancer()

#Creating the features from breast_cancer dataset
features=pd.DataFrame(data.data,columns=data.feature_names)

#creating the target variable from breast_cancer dataset
y =data.target

#Here I am not doing any preprocessing, as the main intention of this code is to show differen ways of building model using KNN
#checking the datatypes of features
features.dtypes
#Creating KNN from scratch
    #1.)Calculate the distance of new point from other data pointds in training data
    #2.) Sort the values based on the distance in ascending order
    #3.) Get the K nearest neighbours
    #4.) choose the maximum labels out of k nearest neighbours

#step 1 is to calculate the distance . Created a function to calcualte the distance between two points
def distance_calculation(x,y,p):
    #x=test point, y=train point , p= power p=1=Manhattan distance, p=2=Euclidian distance
    #calculate the length dimensions (features) of the point
    dimensions=len(x)
    #intializing the distance
    distance=0
    for i in range(dimensions):
        distance+=abs(x[i]-y[i])**2
    distance=distance**(1/p)
    return distance

# split the data in to train and test
X_Train,X_Test,Y_Train,Y_Test=train_test_split(features,y,test_size=0.2,random_state=40)

# Standardizing the data
standardizer=StandardScaler()
X_train = standardizer.fit_transform(X_Train)
X_test = standardizer.transform(X_Test)

#Function to predict the output for the new input
def knn_predictions(X_train,X_test,Y_Train,Y_Test,k,p):
    y_pred=[]
    #predictions for each test point
    for i in X_test:
        #to store distance of all points with the test point
        distances=[]
        for j in X_train:
            distance=distance_calculation(i,j,p)
            distances.append(distance)
        #Converting list to ndarray
        distance_array=np.array(distances)
        #sorting the distances and extracting only nearest k points
        sorted_array=np.argsort(distance_array)[:k]
        #calculating the frequency of nerest neighbours
        counter=Counter(Y_Train[sorted_array])
        #get the nearest label
        label=counter.most_common()[0][0]
        #appending predicted label to another list
        y_pred.append(label)
    return y_pred
 
 #Predict the output labels for the test dataset when k=5 and using euclidian distance
test_predictions=knn_predictions(X_train,X_test,Y_Train,Y_Test,5,2)

#Testing the accuracy by varying k and distance metrics
Accuracy=[]
#Iterating through k values ranging from 1 to 30 and p value 1 and 2
for k in range(1,30):
    for p in range(1,3):
        #calling KNN function to get the predictions
        test_pred=knn_predictions(X_train,X_test,Y_Train,Y_Test,k,p)
        #generationg the accuracy
        accuracy=accuracy_score(Y_Test,test_pred)
        data=k,p,accuracy
        #Appending the accuracy for every k and p value to Accuracy list
        Accuracy.append(data)


#Implementing KNN using sklearn

from sklearn.neighbors import KNeighborsClassifier

KNN_calssifier = KNeighborsClassifier(n_neighbors=7, p=2)
KNN_calssifier.fit(X_train, Y_Train)
y_pred_test = KNN_calssifier.predict(X_test)

print(accuracy_score(Y_Test, y_pred_test))

#Implementing Weighted KNN using sklearn

from sklearn.neighbors import KNeighborsClassifier
#Add weights parmaeter to use weighted KNN
KNN_calssifier = KNeighborsClassifier(n_neighbors=7, weights="distance",p=2)
KNN_calssifier.fit(X_train, Y_Train)
y_pred_test = KNN_calssifier.predict(X_test)

print(accuracy_score(Y_Test, y_pred_test))