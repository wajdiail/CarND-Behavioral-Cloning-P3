# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_model.png "Model"
[image2]: ./examples/original_img.png "original_img"
[image3]: ./examples/crop_img.png "crop_img"
[image4]: ./examples/before_sheer.png "before_sheer"
[image5]: ./examples/after_sheer.png "after_sheer"
[image6]: ./examples/before_flip.png "before_flip"
[image7]: ./examples/after_flip.png "after_flip"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model employed is from the Nvidia's end to end learning for self driving cars [paper][https://arxiv.org/pdf/1604.07316v1.pdf]
The model includes ELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

Image augmentation is used to further reduce the overfitting and make model generalize well.

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases,

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 



Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps in the correct direction and two laps in the wrong direction. I also captured data recovery from left to center and right to center.

After the collection process, I then preprocessed this data by using the image augmentaiton pipeline. 

##### Image augmentation pipeline:

My image augmentation pipeline consist of four steps:

**Image cropping:**
The image was cropped remove the unwanted to top and botton portion that would distract the model more. 

Original image
![alt text][image2]

Cropped image
![alt text][image3]

**Image resizing:**
The cropped image was further resized to half of the orignal size. This reduced the number of parameters and helped train the model fast.

**Image sheering:**
The above reized cropped image is further sheered.

Orginal image
![alt text][image4]

Sheered image
![alt text][image5]

**Image flipping:** 
Finally the image is flipped.

Orignial image
![alt text][image6]

Flipped image
![alt text][image7]


I finally randomly shuffled the data set.

I also used python generator instead of storing the data in the memory. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30. I used an adam optimizer so that manually training the learning rate wasn't necessary.
