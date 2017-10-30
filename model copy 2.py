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
from keras.preprocessing.image import ImageDataGenerator


INPUT_SHAPE = (160, 320, 3)
BATCH_SIZE = 64
EPOCH = 40
PATH_TO_IMG = 'drive_data/IMG/'
PATH_TO_CSV = 'drive_data/driving_log.csv'
CORRECTION = 0.25

def get_csv():
    samples = []
    with open(PATH_TO_CSV) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    return samples

def random_select_image(batch_sample):
    
    random = np.random.randint(4)
    if random == 0:
        name = PATH_TO_IMG+batch_sample[1].split('/')[-1]
        image = cv2.imread(name)
        angle = float(batch_sample[4])+CORRECTION
    elif random == 2:
        name = PATH_TO_IMG+batch_sample[2].split('/')[-1]
        image = cv2.imread(name)
        angle = float(batch_sample[5])-CORRECTION
    else:
        name = PATH_TO_IMG+batch_sample[0].split('/')[-1]
        image = cv2.imread(name)
        angle = float(batch_sample[3])
    
    return image, angle

def get_training(samples, batch_size=32):
    num_samples = len(samples)
    for offset in range(0, num_samples, batch_size):
        batch_samples = samples[offset:offset+batch_size]
        images = []
        angles = []
        for batch_sample in batch_samples:
            result = random_select_image(batch_sample)
            image = result[0] #preprocess_img(result[0])
            random = np.random.randint(3)
            if random == 0:
                image_flipped = np.fliplr(image)
                measurement_flipped = -result[1]
                images.append(image_flipped)
                angles.append(measurement_flipped)
            else:
                images.append(image)
                angles.append(result[1])

        X_train = np.array(images)
        y_train = np.array(angles)
        return sklearn.utils.shuffle(X_train, y_train)

def get_validation(samples):
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
    
    return sklearn.utils.shuffle(X_valid, y_valid)

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

samples = get_csv()
shuffle(samples)
training_data, validation_data = train_test_split(samples, test_size=0.2)
X_train, y_train = get_training(training_data, batch_size=BATCH_SIZE)
X_valid, y_valid = get_validation(validation_data)
gen_train = ImageDataGenerator(height_shift_range=0.2)
gen_valid = ImageDataGenerator()
model = get_model()
samples_per_epoch_train = int(len(X_train) / BATCH_SIZE) * BATCH_SIZE
samples_per_epoch_valid = int(len(X_valid) / BATCH_SIZE) * BATCH_SIZE
history = model.fit_generator(gen_train.flow(X_train, y_train, batch_size=BATCH_SIZE), samples_per_epoch= samples_per_epoch_train, validation_data=gen_valid.flow(X_valid, y_valid, batch_size=BATCH_SIZE), nb_val_samples=samples_per_epoch_valid, nb_epoch=EPOCH)
model.save('model.h5')

    

