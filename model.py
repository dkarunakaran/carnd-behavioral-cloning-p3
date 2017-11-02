from keras.models import Sequential, Model 
from keras.layers import Lambda, Cropping2D, Convolution2D, ELU, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from random import shuffle
import os
import cv2
import numpy as np
import pandas as pd

# Paths
PATH_TO_IMG = 'drive_data/IMG/'
PATH_TO_CSV = 'drive_data/driving_log.csv'

# Shape
INPUT_SHAPE = (160, 320, 3)

# Hyperparameteres
BATCH_SIZE = 32
EPOCH = 15
CORRECTION = 0.25
LEARNING_PARAMETER = .0001 #1e-4

# Get data from csv
def get_csv():
    df = pd.read_csv(PATH_TO_CSV, index_col=False)
    df.columns = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']
    df = df.sample(n=len(df))

    return df

# Randomly selecting the let, right, and center images
def random_select_image(data, i):
     
    random = np.random.randint(3)
    if random == 0:
        path = PATH_TO_IMG+data['left'][i].split('/')[-1]
        difference = CORRECTION
    elif random == 1:
        path = PATH_TO_IMG+data['center'][i].split('/')[-1]
        difference = 0 
    elif random == 2:
        path = PATH_TO_IMG+data['right'][i].split('/')[-1]
        difference = -CORRECTION
        
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    angle = float(data['steer'][i])+difference
  
    return image, angle

# Getting trans images
def trans_image(image, steer):
    trans_range = 100
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 0
    M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, M, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    
    return image_tr, steer_ang

# Getting brightnessed image
def brightnessed_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * random_bright
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    
    return image

# Flipping the images
def flip_img_angle(image, angle):
    image = cv2.flip(image, 1)
    angle *= -1.0

    return image, angle

# Getting fetatures and lables from training and validation data
def get_data(data):
    images = []
    angles = []
    for i in data.index:
        image, angle = random_select_image(data, i)

        # Data augumentation
        if np.random.uniform() < 0.5:
            image, angle = flip_img_angle(image, angle)
        image = brightnessed_img(image)
        image, angle = trans_image(image, angle)
        images.append(image)
        angles.append(angle)

    # Creating as numpy array
    X = np.array(images)
    y = np.array(angles)

    return X, y

# Creating the model
def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.-0.5,input_shape=INPUT_SHAPE))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(1))

    adam = Adam(lr=LEARNING_PARAMETER)
    model.compile(optimizer=adam,loss='mse')

    return model

# Getting data from CSV
samples = get_csv()

# Training and Validation data
training_count = int(0.8 * len(samples))
training_data = samples[:training_count].reset_index()
validation_data = samples[training_count:].reset_index()

# Getting features and labels for training and validation.
X_train, y_train = get_data(training_data)
X_valid, y_valid = get_data(validation_data)

# Instantiating ImageDataGenerator other than yield function
gen_train = ImageDataGenerator(height_shift_range=0.2)
gen_valid = ImageDataGenerator()

# Model using Keras
model = get_model()

# Calculating the no.of sample per epoch for taining and validation.
samples_per_epoch_train = int(len(X_train) / BATCH_SIZE) * BATCH_SIZE
samples_per_epoch_valid = int(len(X_valid) / BATCH_SIZE) * BATCH_SIZE

# Training the model
model.fit_generator(gen_train.flow(X_train, y_train, batch_size=BATCH_SIZE), samples_per_epoch= samples_per_epoch_train, validation_data=gen_valid.flow(X_valid, y_valid, batch_size=BATCH_SIZE), nb_val_samples=samples_per_epoch_valid, nb_epoch=EPOCH)
model.save('model.h5')

    

