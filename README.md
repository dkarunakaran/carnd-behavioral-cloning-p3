# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

This is the third project of the Udacity Selfdriving Car nanodegree.The project is about to train the car to almost go around the track. I have used modified NVIDIA architecture and different data augumentation technique to train the model.

The project has following files: 
* model.py
* drive.py
* model.h5
* video.mp4 (a video recording of your vehicle driving autonomously around the track)


## Model Architecture

I have inspired from NVIDIA architure and made some small changes to the architecture to use it for this project.

<img src="images/cnn-architecture.png" />

The main model structure is given below:

```
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
```

The major differences are:
* Model's input image dimension is (160,320,3) compared to Nvidia model input dimension. 
* Removed one fully connected layer

## Random selection and Data augumenation

I have used 4 data augumenation technique and randomly select left & right images with adjusted steering angles to train the model.

### Random selection

In order to avoid the overfitting, left and right images are randomly selected and adjusted their andle as if it was on the centre. During the autonomus testing, center image is only considered. This is the reason why if the left or right images are selected, then adjusting the steering angle. CORRECTION value is found out by trian and error method and best suited value for this model is .25.

```
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
```
### Data augumenation

One of the first technique is to convert BGR format to RGB

```
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

<img src="images/BGR.png">

<img src="images/RGB.png">


The second technique is flipping the image and change the angle. i.e if an angle is positive, flipping will change the angle to negative and vice versa.
```
# Flipping the images
def flip_img_angle(image, angle):
    image = cv2.flip(image, 1)
    angle *= -1.0
```
Actual image<br/>
<img src="images/actual_image.png">

Actual angle: -0.3012811

Flipped image
<img src="images/flipped_image.png">

Flipped angle: 0.3012811
