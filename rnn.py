# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 12:17:32 2018
@author: Arpit Jayesh Vora
@course: @superDataScience by Kirill and Hadelin
"""
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#keras libraries
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

#importing dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values
                                 
# feature scaling 
sc=MinMaxScaler(feature_range=(0,1))                                 
training_set_scaled=sc.fit_transform(training_set)  

#creating training data set with 60 timesteps
x_train=[];y_train=[]
for i in range(60,len(training_set_scaled)):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

#reshaping
x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# training model
# initialization
regressor = Sequential()
# adding LSTM layers with dropout regularization
regressor.add(LSTM(units=80, return_sequences=  True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# adding LSTM layers with dropout regularization
regressor.add(LSTM(units=80, return_sequences=  True))
regressor.add(Dropout(0.2))
# adding LSTM layers with dropout regularization
regressor.add(LSTM(units=80, return_sequences=  True))
regressor.add(Dropout(0.2))
# adding LSTM layers with dropout regularization
regressor.add(LSTM(units=80, return_sequences=  True))
regressor.add(Dropout(0.2))
# adding LSTM layers with dropout regularization
regressor.add(LSTM(units=80))
regressor.add(Dropout(0.2))
# adding a fully connected layer 
regressor.add(Dense(units=1))
# compiling
regressor.compile(optimizer='adam', loss='mean_squared_error')
# Keras callbacks
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')
 
history = regressor.fit(x_train, y_train, shuffle=True, epochs=100,
                        callbacks=[es, rlr,mcp, tb], validation_split=0.2, verbose=1, batch_size=64)
 
# Creating Test data
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# prediction
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
