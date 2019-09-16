# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center_2018_08_05_08_56_56_469.jpg "Center Camera Image"
[image3]: ./examples/left_2018_08_05_08_56_56_469.jpg "Left Camera Image"
[image4]: ./examples/right_2018_08_05_08_56_56_469.jpg "Right Camera Image"
[image5]: ./examples/crop.jpg "Recovery Image"
[image6]: ./examples/filp_left_2018_08_05_08_56_56_469.jpg "Flipped Image"
[image7]: ./examples/filp_right_2018_08_05_08_56_56_469.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 68-72) 

The model includes RELU layers to introduce nonlinearity (code line 68-72, and the data is normalized in the model using a Keras lambda layer (code line 66). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 73). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 85). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 91).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road with the camera correction 0.2

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia's  end to end learning for self-driving  cars , I thought this model might be appropriate because the network used for training a real car to drive autonomously.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that the mean squared error on the validation set have decreased. I tried to get more driving data and use dropout method.

Then I use adam optimizer to optimize the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I augmented the dataset with flipping the image and reverse driving round to collect more data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 64-80) consisted of a convolution neural network with the following layers and layer sizes 

![alt text][image1]

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle can be trained from multiple cameras.

![alt text][image3]

![alt text][image4]

These images show what a recovery looks like starting from the hood of the car, and the top of the images captures are trees and hills and sky. In order to train faster I crop each image to focus on only the portion of the image that is useful for predicting a steering angel. I crop the image with 70 rows pixels from the top and 25 rows pixels from the bottom of the image.

![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would add more data for training and for a better result.

 For example, here is an image that has then been flipped:

![alt text][image6]

![alt text][image7]



After the collection process, I had 1266 number of data points. I then preprocessed this data by normalizing, I use keras lambda to handle it. code as belows:

`model.add(Lambda(lambda x: x / 255.0 - 0.5, intput_shape=(160, 320, 3)))`


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

The ideal number of epochs was 5 as evidenced by loss value.

I used an adam optimizer so that manually training the learning rate wasn't necessary.