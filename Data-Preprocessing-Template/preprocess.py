# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 12:43:31 2017

@author: gilgamesh
"""
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset=pd.read_csv('Data.csv')
dataset.head()

#creating the features and targets
x= dataset.iloc[:,0:3].values
#print(x)
y= dataset.iloc[:,-1].values
#print(y)

#handling missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
print (x)

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
#one hot encoding independent variables
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(x)

# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)