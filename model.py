from keras.models import Sequential, Model 
from keras.layers import Lambda, Cropping2D, Convolution2D, ELU, Flatten, Dense
import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from random import shuffle

input_shape = (160, 320, 3)

def get_csv():
    samples = []
    with open('./drive_data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    return samples

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = np.empty([batch_size, input_shape[0],input_shape[1], input_shape[2]])
            angles = np.empty([batch_size, 1])
            for i, batch_sample in enumerate(batch_samples):
                name = './drive_data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images[i] = center_image
                angles[i] = center_angle

            # trim image to only see section with road
            #print(images.shape)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def get_model():
    model = Sequential()

    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=input_shape, output_shape=input_shape))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init="glorot_uniform"))

    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init="glorot_uniform"))

    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init="glorot_uniform"))

    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="glorot_uniform"))

    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="glorot_uniform"))

    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164,init="glorot_uniform"))

    model.add(ELU())
    model.add(Dense(100,init="glorot_uniform"))

    model.add(ELU())
    model.add(Dense(50,init="glorot_uniform"))

    model.add(ELU())
    model.add(Dense(10,init="glorot_uniform"))

    model.add(ELU())
    model.add(Dense(1,init="glorot_uniform"))

    return model

samples = get_csv()
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = get_model()
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
print(history)

    

