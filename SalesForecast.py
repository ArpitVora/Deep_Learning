# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 12:15:32 2018
@author: Arpit Jayesh Vora
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
#Keras library 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

# pre processing
# Load training data set from CSV file
training_data_df = pd.read_csv("sales_data_training.csv")

# Load testing data set from CSV file
test_data_df = pd.read_csv("sales_data_test.csv")

# Data needs to be scaled to a small range like 0 to 1
sc = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
scaled_training = sc.fit_transform(training_data_df)
scaled_testing = sc.transform(test_data_df)

# Print scaler
print("Note: scaled multiplication {:.10f} and addition {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

# making it scaled data as dataframe
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

# Saving as CSV files
scaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)
scaled_testing_df.to_csv("sales_data_testing_scaled.csv", index=False)

#importing dataset
datasetTrain  = pd.read_csv('sales_data_training_scaled.csv')
datasetTest  = pd.read_csv('sales_data_testing_scaled.csv')

# feature scale
x=datasetTrain.drop('total_earnings', axis=1).values
y=datasetTrain[['total_earnings']].values                   

# training model
# initialization of sequential model
classifier = Sequential()              
# adding hidden layers
classifier.add(Dense(units=50,activation='relu',kernel_initializer="glorot_uniform", input_dim=9, name="layer_1"))
classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_2"))
classifier.add(Dropout(0.2))
#trying with extra hidden layer
classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_3"))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_4"))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_5"))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_6"))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_7"))
classifier.add(Dropout(0.2))
#adding output layer
classifier.add(Dense(units = 1, activation = 'linear', kernel_initializer='glorot_uniform',name = "layer_8"))
# compiling the network
classifier.compile(optimizer='adam',loss='mean_squared_error')
# visualize on tensor board
logger=keras.callbacks.TensorBoard(log_dir='./logs',histogram_freq=0,write_graph=True)
classifier.fit(x,y,batch_size=10,epochs=128,callbacks=[logger])

# predict method
X_test = datasetTest.drop('total_earnings', axis=1).values
Y_test = datasetTest[['total_earnings']].values
y_pred=classifier.predict(X_test)
classifier.evaluate(X_test,Y_test)

#read test data set 2
test_data_df = pd.read_csv("sales_data_test_scaled.csv")
#prediction for test data
X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values
y_pred=classifier.predict(X_test)
X = pd.read_csv("proposed_new_product.csv").values
test_error_rate = classifier.evaluate(X_test, Y_test)

# prediction for new values 
prediction = classifier.predict(X)
# re scaling the predicted value to original value
prediction = sc.inverse_transform(prediction)
print("Earnings Prediction - ${}".format(prediction))

# parameter tuning with cross validation
def buildClassifier():
    classifier = Sequential()              
    classifier.add(Dense(units=50,activation='relu',kernel_initializer="glorot_uniform", input_dim=9, name="layer_1"))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_2"))
    classifier.add(Dropout(0.2)) 
    # adding extra dense layer
    classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_3"))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_11"))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_22"))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_33"))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_4"))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_5"))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_6"))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 50, activation = 'relu', kernel_initializer='glorot_uniform',name = "layer_7"))
    classifier.add(Dropout(0.2))
    #adding output layer
    classifier.add(Dense(units = 1, activation = 'tanh', kernel_initializer='glorot_uniform',name = "layer_8"))
    # compiling network
    classifier.compile(optimizer='adam',loss='mean_squared_error')
    return classifier

classifier = KerasClassifier(build_fn = buildClassifier, batch_size = 10,epochs=100)
accu = cross_val_score(estimator = classifier,X=x,y=y,cv=10,n_jobs=1)
accu_mean=accu.mean()
accu_var=(accu.std())**2
classifier.evalu
         
classifier = KerasClassifier(build_fn = buildClassifier)
parameteres = {'batch_size' : [10,20,30,40,50],
                'epochs' : [25,50,100,150]}

grid_search = GridSearchCV(estimator=classifier,param_grid = parameteres,scoring='accuracy',cv=5)
grid_search=grid_search.fit(x,y)
best_param=grid_search.best_params_
best_accu=grid_search.best_score_
