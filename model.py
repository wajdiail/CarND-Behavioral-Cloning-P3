
#Importing Dependencies

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import glob as glob
import csv
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Lambda, Convolution2D, Activation, Cropping2D, Dropout
from skimage import transform as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import random


 def extract_camera_pos(lines, camera_pos):
    
    X,y=[],[]
    
    if camera_pos=='center':
        pos=0
        correction=0
    if camera_pos=='left':
        pos=1
        correction=0.2
    if camera_pos=='right':
        pos=2
        correction=-0.2
        
    for i in range(0, len(lines)):
        X.append(lines[i][pos])
        y.append(float(lines[i][3])+correction)
        
    return X,y    


def data_loading():
    
    with open('./examples/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        lines = []
        next(reader)
        for row in reader:
            lines.append(row)
    
    X_center, y_center = extract_camera_pos(lines, camera_pos='center')    
    X_left, y_left = extract_camera_pos(lines, camera_pos='left')
    X_right, y_right = extract_camera_pos(lines, camera_pos='left')
    
    #merging all position into single list
    X_all, y=[], []
    X_all.extend(X_center)
    X_all.extend(X_left)
    X_all.extend(X_right)
    y.extend(y_center)
    y.extend(y_left)
    y.extend(y_right)
    

    #modifying the absolute path to relatvie path
    #test_img =[]
    X=[]
    for i in range(len(X_all)):
        filename = X_all[i].split('/')[-1]
        path = './examples/IMG/'+ filename
        X.append(path)
        #test_img.append(mpimg.imread(path))
        
    return X, y

def image_augmentation(img_list, steering_angle):
    
    #Cropping image
    crop_img=[]
    for i in range(len(img_list)):
        crop_img.append(img_list[i][60:140])
    
 
        
    #Resizing image
    resized_img=[]
    for i in range(len(crop_img)):
        resized_img.append(cv2.resize(crop_img[i],None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))
    
    #Image Sheering
    sheared_img=[]
    #print(len(steering_angle))
    #print(len(resized_img))
    
    for i in range(0, len(resized_img), 3): 
        rand_value = round(random.uniform(-1,1),1)
        #Create Afine transform
        afine_tf = tf.AffineTransform(shear=rand_value)
        # Apply transform to image data
        resized_img.append(tf.warp(resized_img[i], inverse_map=afine_tf))
        steering_angle.append(steering_angle[i])

    
    y_steer = []
    for item in steering_angle:
        y_steer.append(float(item))


    #Image flipping
    X_img = np.array(resized_img)
    y_steer =np.array(y_steer)
    

        
    #shuffle(x_img, y_steer)
    
    for i in range(0, len(X_img), 3):
        X_img[i]= np.fliplr(X_img[i])
        y_steer[i]= -y_steer[i]
                           
    return X_img, y_steer


def image_generator(img_path_list, steering_angle, batch_size):
    
    num_of_samples=len(img_path_list)
    
    while True:
        
        for offset in range(0, num_of_samples, batch_size):

            x_img_batch, y_steer_batch = img_path_list[offset:offset+batch_size], steering_angle[offset:offset+batch_size]
        
            img=[]
            
            for i in range(len(x_img_batch)):
                path = x_img_batch[i]
                img.append(mpimg.imread(path))
            
            X_img, y_steer = image_augmentation(img_list=img, steering_angle = y_steer_batch)
                
            yield shuffle(X_img, y_steer)
        

def train_data_generator(batch_size=128):
        return image_generator(img_path_list=X_train, steering_angle=y_train, batch_size=batch_size)

def valid_data_generator(batch_size=128):
        return image_generator(img_path_list=X_valid, steering_angle=y_valid, batch_size=batch_size,)

#Data loading
X,y = data_loading()  

#Splitting the data into training  and validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

#Defining the model in Keras

model = Sequential()
model.add(Lambda(lambda x:x/127.5-1.0, input_shape=(40,160,3)))
model.add(Convolution2D(24,(5,5), strides=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(36,(5,5),strides=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(48,(5,5)))
model.add(Activation('elu'))
model.add(Convolution2D(64,(3,3)))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dense(10))
model.add(Activation('elu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

#Running the model
model.fit_generator(generator=train_data_generator(2000), epochs=30, steps_per_epoch =5, validation_data=valid_data_generat(2000), validation_steps=3)
model.save('model.h5')
#model.summary()


