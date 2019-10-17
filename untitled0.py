# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:54:01 2019

@author: sridhar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
from sklearn.cross_validation import train_test_split


#label encoder 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#X1 = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)
y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values = X, axis =1)