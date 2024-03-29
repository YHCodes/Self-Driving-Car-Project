# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./output/vis_images.png "Visual datasets"
[image2]: ./output/distribution.png "Distribution"
[image3]: ./output/gray.png "gray image"
[image4]: ./output/normalized.png "Normalized"
[image5]: ./output/sign_names.png "Sign Names"
[image6]: ./output/augment.png "Augmentation"
[image7]: ./output/test_imgs.png "Test Images"
[image8]: ./output/softmax.png "soft max"
[image9]: ./output/top5.png "top5"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
  * 34799
* The size of the validation set is ?
  * 4410
* The size of test set is ?
  * 12630
* The shape of a traffic sign image is ?
  * (32, 32, 3)
* The number of unique classes/labels in the data set is ?
  * 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set and sign names table.

![alt text][image1]

![alt text][image5]

Here shows data set distribution.

![alt text][image2]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it can make training set smaller and make train faster.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data because it can prevent overfitting and increase the accuracy of the model.

Here is an example of a traffic sign image before and after normalizing.

![alt text][image4]

I decided to generate additional data because the data set distributions are non-uniformity.

To add more data to the the data set, I used the following techniques:

- random bright image 
- random rotate image
- random translate image
- random shear image

Here is an example of an original image and an augmented image:

![alt text][image6]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|      Layer      |                  Description                   |
| :-------------: | :--------------------------------------------: |
|      Input      |               32x32x1 Gray image               |
| Convolution 3x3 | 1x1 stride,valid padding, outputs 30 x 30 x 80 |
|      RELU       |                                                |
|    Drop Out     |                keep_prob = 0.5                 |
| Convolution 3x3 |  1x1 stride, same padding, outputs 30x30x120   |
|      RELE       |                                                |
|     MaxPool     |         2x2 stride, outputs 15x15x120          |
|    Drop Out     |                keep_prob = 0.5                 |
| Convolution 4x4 |  1x1 stride, valid padding, outputs 12x12x180  |
|      RELU       |                                                |
|    DROP OUT     |                keep_prob = 0.5                 |
| Convolution 3x3 |  1x1stride, same padding,  outputs 12x12x200   |
|      RELU       |                                                |
|    MAX POOL     |          2X2 stride, outputs 6x6x200           |
|    DROP OUT     |                keep_prob = 0.5                 |
| Convolution 3x3 |   1x1stride, valid padding, outputs 4x4x200    |
|      RELU       |                                                |
|    MAX POOL     |          2x2 stride, outputs 2x2x200           |
|    Drop out     |                keep_prob = 0.5                 |
| Fully Connected |                  outputs 800                   |
| Fully Connected |                   outputs 80                   |
| Fully Connected |                   outputs 80                   |
| Fully Connected |                   outputs 43                   |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters:

- type op optimizer: `AdamOptimizer`
- batch size = 128
- epochs = 25
- learning rate = 0.0001
- keep_prob = 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
  * 99.8%
* validation set accuracy of ? 
  * 98.3%
* test set accuracy of ?
  * 97.05%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * LeNet, Convolutional Network architecture is good at handle image feature extraction.
* What were some problems with the initial architecture?
  * the initial architecture is  easy to over-fitting
  * Dataset may  be uneven distribution
* Which parameters were tuned? How were they adjusted and why?
  * weights and bias. they adjusted to make reduce_mean smallest and make predict accuracy highest.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * convolution layer work well with this problem because convolutional is good at extracting image feature.
  * dropout layer:  can prevent overfitting.
  * MaxPool: Reduce parameter and speed up training.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7]  


The thid and the fourth image might be difficult to classify because the image looks very fuzzy.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| No entry | No entry  |
| Keep Right | Keep Right |
| Stop	| End of no passing	|
| Speed limit (30km/h)	| Speed limit(30 km/h)	|
| Road work	| Road work    |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a No entry sign (probability of 1), and the image does contain a no entry sign. The top five soft max probabilities were:

![alt text][image9]



**softmax probabilities**

![alt text][image8]



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


