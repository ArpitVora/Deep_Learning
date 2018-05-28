# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:47:03 2018
@author: Arpit Jayesh Vora
@course: @superDataScience by Kirill and Hadelin
"""

from keras import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#creating sequential object
classifier = Sequential()

#creating convolution layer
classifier.add(Conv2D(32,(3,3), input_shape =(64,64,3), activation = 'relu'))

#creating MaxPooling layer, by default the value of stride here is 2 as pool size is 2*2
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# creating flattening layer
classifier.add(Flatten())

# creating Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#compiling cnn
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# image pre processing and fit method
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# data augmentation
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/32,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)


#predict single image
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
classifier.predict(test_image)
training_set.class_indices