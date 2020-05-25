# **Udacity Behavioral Cloning Project** 
---

Note: This project makes use of  [Udacity Driving Simulator](https://github.com/udacity/self-driving-car-sim) and Udacity Workspace.

[//]: # (Image References)

[image1]: ./examples/model_architecture.png
[image2]: ./examples/nvidia-architecture.png "NVIDIA Architecture"
[image3]: ./examples/original.jpg "Original Image"
[image4]: ./examples/flipped.jpg "Flipped Image"
[image5]: ./examples/left.jpg "Left Image"
[image6]: ./examples/center.jpg "Center Image"
[image7]: ./examples/right.jpg "Right Image"
[image8]: ./examples/run1.jpg


## Introduction


The main objectives of this project are:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

The five files for this project are: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (u are reading it)
* run1.avi (a video recording of your vehicle driving autonomously around the track for at least one full lap)

In this project, I used what I've learned about deep neural networks and convolutional neural networks to clone driving behavior. I trained, validatde and tested a model using Keras. The model will output a steering angle to an autonomous vehicle.

These collected data are then fed into neural networks to output the correct steering angle of the car. 

The neural network model is saved as `model.h5`. 
## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.


The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

## Network Architecture

The neural network architecture used for this project is a known self-driving car model from [NVIDIA](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).
<div align=center>![alt text][image2]

The detialed imformation for the network and the input/output shape for each layer are shown.
![alt text][image1]

## Data Collection and Data Processing

Udacity has provided sets of sample data which are enough to produce a working model. Only the provided data is used in this project.

![alt text][image3]
![alt text][image4]

_Original Image & Flipped Image_

The left and right camera images are also used to train the network.
The steering angle for both left and right images are adjusted by +0.3 for the left frame and -0.3 for the right. Neural network require large amount of data to produce better model. The images are flipped and the steering angle is mutiplied by -1 to achieve data augmaentaion.

![alt text][image5]
![alt text][image6]
![alt text][image7]

_Left, Center, and Right Dashboard Camera Images_



## Fine-Tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Network layers could have been modified to get better peroformance.
Before training the model, the dataset is shuffled and split into 90% for training and 10% for validation. Testing data is not generated as the testing is runing the driving simulator with the model.

Attempts to reduce overfitting in the model:
The model contains three dropout layers in order to reduce overfitting.


## Result

The final video is named as run1.avi. It has performed well to me.

![alt text][image8]
