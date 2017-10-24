from keras.models import Sequential, Model 
from keras.layers import Lambda, Cropping2D, Convolution2D, ELU, Flatten, Dense


def nvidia_architecture():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1, input_shape=(66,200,3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))
    
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))

    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))

    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="he_normal"))

    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="he_normal"))

    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164,init="he_normal"))

    model.add(ELU())
    model.add(Dense(100,init="he_normal"))

    model.add(ELU())
    model.add(Dense(50,init="he_normal"))

    model.add(ELU())
    model.add(Dense(10,init="he_normal"))

    model.add(ELU())
    model.add(Dense(1,init="he_normal"))


    return model

nvidia_architecture()
    

