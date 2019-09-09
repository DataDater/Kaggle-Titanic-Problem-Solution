# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:10:27 2019

@author: Ayan Dasgupta
"""
#Importing packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Forming Dataset
dataset = pd.read_csv('Train_Test.csv')
X = dataset.iloc[:, [2,4,5,6,7,8,10, 12, 13]].values
y = dataset.iloc[:, 1].values
y = np.reshape(y, (1309, 1))

#Encoding Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Title = LabelEncoder()
X[:, 1] = labelencoder_Title.fit_transform(X[:, 1])
labelencoder_Sex = LabelEncoder()
X[:, 2] = labelencoder_Sex.fit_transform(X[:, 2])
labelencoder_Cabin = LabelEncoder()
X[:, 7] = labelencoder_Cabin.fit_transform(X[:, 7])
labelencoder_Embarked = LabelEncoder()
X[:, 8] = labelencoder_Embarked.fit_transform(X[:, 8])
onehotencoder = OneHotEncoder(categorical_features = [1, 7, 8])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#Splitting into Training and Test Sets
X_train = X[0:891, :]
y_train = y[0:891, :]
X_test = X[891:, :]
y_test = y[891:, :]

#Scaling the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#Developing the ANN
classifier = Sequential()
classifier.add(Dense(output_dim = 18, init = 'uniform', activation = 'relu', input_dim = 35))
classifier.add(Dense(output_dim = 18, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Writing to csv
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv("prediction.csv")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)