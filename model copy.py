from keras.models import Sequential, Model 
from keras.layers import Lambda, Cropping2D, Convolution2D, ELU, Flatten, Dense, SpatialDropout2D, Dropout
from keras.optimizers import Adam
import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from random import shuffle

INPUT_SHAPE = (160, 320, 3)
BATCH_SIZE = 64
EPOCH = 5
PATH_TO_IMG = 'data/IMG/'
PATH_TO_CSV = 'data/driving_log.csv'
CORRECTION = 0.2

def get_csv():
    samples = []
    with open(PATH_TO_CSV) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    return samples

def random_select_image(batch_sample):
    '''
    random = np.random.randint(4)
    if random == 0:
        name = PATH_TO_IMG+batch_sample[1].split('/')[-1]
        image = cv2.imread(name)
        angle = float(batch_sample[4])+CORRECTION
    elif random == 1:
        name = PATH_TO_IMG+batch_sample[2].split('/')[-1]
        image = cv2.imread(name)
        angle = float(batch_sample[5])-CORRECTION
    else:
        name = PATH_TO_IMG+batch_sample[0].split('/')[-1]
        image = cv2.imread(name)
        angle = float(batch_sample[3])

    #image = preprocess_img(image)
    '''
    name = PATH_TO_IMG+batch_sample[0].split('/')[-1]
    image = cv2.imread(name)
    angle = float(batch_sample[3])
    
    return image, angle

def training_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                result = random_select_image(batch_sample)
                random = np.random.randint(4)
                if random == 0:
                    image_flipped = np.fliplr(result[0])
                    measurement_flipped = -result[1]
                    images.append(image_flipped)
                    angles.append(measurement_flipped)
                else:
                    images.append(result[0])
                    angles.append(result[1])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

    '''
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = np.empty([batch_size, NEW_SHAPE[0], NEW_SHAPE[1], NEW_SHAPE[2]])
            angles = np.empty([batch_size, 1])
            for i, batch_sample in enumerate(batch_samples):
                result = random_select_image(batch_sample)
                random = np.random.randint(4)
                if random == 0:
                    image_flipped = np.fliplr(result[0])
                    measurement_flipped = -result[1]
                    images[i] = image_flipped
                    angles[i] = measurement_flipped
                else:
                    images[i] = result[0]
                    angles[i] = result[1]

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
    '''

def validation_generator(samples):
     while 1:
        shuffle(samples)
        images = []
        angles = []
        for i, sample in enumerate(samples):
            name = PATH_TO_IMG+sample[0].split('/')[-1]
            center_image = cv2.imread(name)
            center_angle = float(sample[3])
            images.append(center_image)
            angles.append(center_angle)

        X_valid = np.array(images)
        y_valid = np.array(angles)
        
        yield sklearn.utils.shuffle(X_valid, y_valid)

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.-0.5, input_shape=INPUT_SHAPE))
    model.add(Cropping2D(cropping=((60, 20), (0, 0)),input_shape=INPUT_SHAPE))
    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1164, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=0.001), loss='mse')

    return model

samples = get_csv()
training_data, validation_data = train_test_split(samples, test_size=0.2)

train_generator = training_generator(training_data, batch_size=BATCH_SIZE)
validation_generator = validation_generator(validation_data)
model = get_model()
samples_per_epoch = int(len(training_data) / BATCH_SIZE) * BATCH_SIZE
history = model.fit_generator(train_generator, samples_per_epoch= samples_per_epoch, validation_data=validation_generator, nb_val_samples=len(validation_data), nb_epoch=EPOCH)
model.save('model.h5')

    

