# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:02:08 2017

@author: gilgamesh
"""

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
dataset.head()

#creating the features and targets
x= dataset.iloc[:,2:4].values
print(x)
y= dataset.iloc[:,-1].values
print(y)

# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#fit logistic regression to train data
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

#predicting for test set
y_pred=classifier.predict(x_test)
print(y_pred)

#create a confusion matrix
from sklearn.metrics import  confusion_matrix
cm=confusion_matrix(y_test,y_pred)
acc=(cm[1,1]+cm[0,0])
print(acc)
