from keras.models import Sequential, Model 
from keras.layers import Lambda, Cropping2D, Convolution2D, ELU, Flatten, Dense, SpatialDropout2D, Dropout
import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from random import shuffle

INPUT_SHAPE = (160, 320, 3)
BATCH_SIZE = 64
EPOCH = 40
PATH_TO_IMG = 'drive_data/IMG/'
PATH_TO_CSV = 'drive_data/driving_log.csv'
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
    elif random == 1 or random == 3:
        name = PATH_TO_IMG+batch_sample[0].split('/')[-1]
        image = cv2.imread(name)
        angle = float(batch_sample[3])
    else:
        name = PATH_TO_IMG+batch_sample[2].split('/')[-1]
        image = cv2.imread(name)
        angle = float(batch_sample[5])-CORRECTION

    '''
    name = PATH_TO_IMG+batch_sample[0].split('/')[-1]
    image = cv2.imread(name)
    angle = float(batch_sample[3])

    return image, angle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = np.empty([batch_size, INPUT_SHAPE[0],INPUT_SHAPE[1], INPUT_SHAPE[2]])
            angles = np.empty([batch_size, 1])
            count = 0
            for i, batch_sample in enumerate(batch_samples):
                result = random_select_image(batch_sample)
                #image_flipped = np.fliplr(image)
                #measurement_flipped = -measurement
                images[i] = result[0]
                angles[i] = result[1]

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def get_model():
    model = Sequential()

    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=INPUT_SHAPE, output_shape=INPUT_SHAPE))
    #model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=INPUT_SHAPE))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(ELU())
    model.add(SpatialDropout2D(0.2))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(ELU())
    model.add(SpatialDropout2D(0.2))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(ELU())
    model.add(SpatialDropout2D(0.2))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(SpatialDropout2D(0.2))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(100))
    model.add(ELU())

    model.add(Dense(50))
    model.add(ELU())

    model.add(Dense(10))
    model.add(ELU())

    model.add(Dense(1))
    model.add(Dropout(0.5))

    return model

samples = get_csv()
training_data, validation_data = train_test_split(samples, test_size=0.2)

train_generator = generator(training_data, batch_size=BATCH_SIZE)
validation_generator = generator(validation_data, batch_size=BATCH_SIZE)

model = get_model()
model.compile(loss='mse', optimizer='adam')
samples_per_epoch = int(len(training_data) / BATCH_SIZE) * BATCH_SIZE
history = model.fit_generator(train_generator, samples_per_epoch= samples_per_epoch, validation_data=validation_generator, nb_val_samples=len(validation_data), nb_epoch=EPOCH)
print(history)
model.save('model.h5')

    

