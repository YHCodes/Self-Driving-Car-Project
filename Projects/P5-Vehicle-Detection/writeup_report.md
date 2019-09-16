## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./examples/sliding_windows.jpg
[image3a]: ./output_images/bottom_half_window.png
[image3b]: ./output_images/more_window.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the **fourth** code cell of the IPython notebook

I started by reading in all the `vehicle` and `non-vehicle` images.  

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:



![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations=15`, `pixels_per_cell=8`, and `cells_per_block=2`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.feature.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=15`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and get the highest test data accuracy of SVC

Finaaly , I set  `orientations=15`,  `pixels_per_cell=8`,  `cells_per_block=2`  , 

get the **test accuracy of SVC = 0.9899**



#### 3. Describe how (and identify where in your code) you trained  a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the eighth code cell of the IPython notebook

I trained a linear SVM using  8792 car images and 8978 not car images, and I use `train_test_split()` to random split 80% training set and 20% testing set.

for every training image,  I extract features include `bin_spatial_features` `color_hist_features` and `hog_features` :

- `bin_spatial()` : I used `cv2.resize()` to resize image size to (32, 32) to get bin_spatial features.
- `color_hist()`: I used `np.histogram()` to compute the histogram of the color channels separately.
- `get_hog_features()` : I used `skimage.feature.hog()` to extract hog features



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]



first, I select interst region of the bottom half image to slide windows, because top bottom  images are sky or other we don't need.

I use `xy_window=(96, 96)` and `xy_overlap=(0.5, 0.5)` to get window bounding boxes, just like.

![alt text][image3A]

I get more candidated sliding window for improving svm classifier and having a better prediction, I used multiple group parameters to get more window to detect.

![alt text][image3B]



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

optimize the performance of my classifier

1. get more data to train
2. Use `sklearn.preprocessing.StandardScaler()` to normalize feature vectors

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

then , I tried to extract features:

- I tried to explore different color space to see which is the better for extracting color features.
- I used `skimage.feature.hog()` to extract hog features.
- I used `cv2.resize()` to reduce spatial to get bin spatial features.

and, I tried to train svm classifier

- use `sklearn.preprocessing.StandardScaler()` to normalize my feature vectors for training svm classifier
- use `sklearn.model_selection.train_test_split()` to randomly split dataset for training and testing.
- use `sklearn.svm.LinearSVC()` to train a svm classfier



problems:

- data set is not enough
-  in some unfamiliar scene it may  detect failed.
- sliding window search is very slow, it's a big problem



What could you do to make it more robust?

1. collect more data
2. tried to use neuro networks to train a bettel model