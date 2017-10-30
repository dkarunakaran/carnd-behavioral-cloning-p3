from keras.models import Sequential, Model 
from keras.layers import Lambda, Cropping2D, Convolution2D, ELU, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
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
BATCH_SIZE = 64
EPOCH = 2
CORRECTION = 0.25

# Get data from csv
def get_csv():
    df = pd.read_csv(PATH_TO_CSV, index_col=False)
    df.columns = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']
    df = df.sample(n=len(df))

    return df

# Randomly selecting the let, right, and center images
def random_select_image(data, i):
    random = np.random.randint(4)
    if random == 0:
        path = PATH_TO_IMG+data['left'][i].split('/')[-1]
        image = cv2.imread(path)
        angle = float(data['steer'][i])+CORRECTION
    elif random == 2:
        path = PATH_TO_IMG+data['right'][i].split('/')[-1]
        image = cv2.imread(path)
        angle = float(data['steer'][i])-CORRECTION
    else:
        path = PATH_TO_IMG+data['center'][i].split('/')[-1]
        image = cv2.imread(path)
        angle = float(data['steer'][i])
    
    return image, angle

# Getting fetatures and lables from training data
def get_training(data):
    images = []
    angles = []
    for i in data.index:
        result = random_select_image(data, i)

        # Rndomly flipping the image
        random = np.random.randint(3)
        if random == 0:
            image_flipped = np.fliplr(result[0])
            measurement_flipped = -result[1]
            images.append(image_flipped)
            angles.append(measurement_flipped)
        else:
            images.append(result[0])
            angles.append(result[1])

    # Creating as numpy array
    X_train = np.array(images)
    y_train = np.array(angles)

    return X_train, y_train

# Getting fetatures and lables from validation data
def get_validation(data):
    images = []
    angles = []
    for i in data.index:
        path = PATH_TO_IMG+data['center'][i].split('/')[-1]
        image = cv2.imread(path)
        angle = float(data['steer'][i])
        images.append(image)
        angles.append(angle)

    # Creating as numpy array
    X_valid = np.array(images)
    y_valid = np.array(angles)
    
    return X_valid, y_valid

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
    model.compile(optimizer='adam', loss='mse')

    return model

# Getting data from CSV
samples = get_csv()

# Training and Validation data
training_count = int(0.8 * len(samples))
training_data = samples[:training_count].reset_index()
validation_data = samples[training_count:].reset_index()

# Getting features and labels for training and validation.
X_train, y_train = get_training(training_data)
X_valid, y_valid = get_validation(validation_data)

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

    

