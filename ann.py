# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:21:09 2018
Title: Customer Churn Prediction
@author: Arpit Jayesh Vora
@course: @superDataScience by Kirill and Hadelin
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#Keras library for ANN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
#importing data set 
dataset  = pd.read_csv('Churn_modelling.csv')
x = dataset.iloc[:,3:13].values
y= dataset.iloc[:,13].values              
                        
#pre processing, kind of factorizing
#for loop is to transform string to categorical values, for example male=1 female=0
for i in range(1,3):
    le_x = LabelEncoder()
    x[:,i]=le_x.fit_transform(x[:,i])

# now using OneHotEncoder, we will create dummy variables
oneHotEncoder = OneHotEncoder(categorical_features= [1])
x = oneHotEncoder.fit_transform(x).toarray()
# to Avoid dummy variable trap, we eliminate the first variable
# if we have m variables then we go ahead with m-1 variables
# we have to avoid MultiColinearity problem
x = x[:,1:]

# Split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#classifier initialization for ANN
classifier = Sequential()
#adding input layer by adding first hidden layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform',input_dim=11,name = "layer_1"))
#adding second hidden layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_2"))
classifier.add(Dropout(0.1))
#trying with extra hidden layer, results are better (83% to 86%)
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_3"))
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_4"))
classifier.add(Dropout(0.1))
#adding output layer
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer='glorot_uniform',name = "layer_5"))
classifier.add(Dropout(0.1))
#compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=['accuracy'])
# tensor flow logger
logger = keras.callbacks.TensorBoard(log_dir='logs/{}'.format("basic run"),histogram_freq=0,write_graph=True)
#to fit model with inputs`
classifier.fit(x_train,y_train,batch_size=10,epochs=25,callbacks=[logger])

#Predicting the results of test data
y_pred=classifier.predict(x_test)
y_pred = (y_pred>0.6)
cfm = confusion_matrix(y_test,y_pred)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
accuracy=((tp+tn)/(tp+tn+fp+fn))*100
print("accuracy = ",accuracy,"%")

#-----------------------------------------------------------------------------------------------------------------
# cross validation

def build_class():
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform',input_dim=11))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer='glorot_uniform'))
    classifier.add(Dropout(0.1))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_class, batch_size = 10,epochs=10)
accu = cross_val_score(estimator = classifier,X=x_train,y=y_train,cv=10,n_jobs=1)
accu_mean=accu.mean()
accu_var=(accu.std())**2
        
#-----------------------------------------------------------------------------------------------------------------

# parameter tuning

def build_class(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform',input_dim=11))
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform'))
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform'))
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform'))
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer='glorot_uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_class)
parameteres = {'batch_size' : [10,20,30,40,50],
                'epochs' : [25,30,40],
                'optimizer' : ['adam','rmsprop']}
grid_search = GridSearchCV(estimator=classifier,param_grid = parameteres,scoring='accuracy',cv=5)
grid_search=grid_search.fit(x_train,y_train)
best_param=grid_search.best_params_
best_accu=grid_search.best_score_
