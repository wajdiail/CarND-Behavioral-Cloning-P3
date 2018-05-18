# **Behavioral Cloning** 

## Writeup Template

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


---

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model employed is from the Nvidia's end-to-end learning for self driving cars [paper](https://arxiv.org/pdf/1604.07316v1.pdf)
The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py code line 158). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py lines 167). 
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 176).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the both sides of the road and from image augmentation pipeline

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model from the Nvidia's end-to-end deeplearning [paper](https://arxiv.org/pdf/1604.07316v1.pdf). The paper does not mention the type of activation unit used. Hence I started with RELUs. The result obtained was good as the car was completing the track but at few places the car was driving on the road lane markers or crossing it. 

After changing the activation unit to ELUs the car was driving in the middle of the road. My initials models had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Dropout layer and Image augmentation is used to further reduce the overfitting and make model generalize well.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 157-176) consisted of a convolution neural network with the following layers and layer sizes 

Input Shape: 40x160x3 <br />
Lamba: Image normalization <br />
Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU <br />
Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU <br />
Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU <br />
Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU <br />
Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU <br />
Drop out (0.5) <br />
Fully connected: neurons: 100, activation: ELU <br />
Fully connected: neurons: 50, activation: ELU <br />
Fully connected: neurons: 10, activation: ELU <br />
Fully connected: neurons: 1 (output) <br />
Optimizer: adam <br />
Loss function: mse <br />

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps in the correct direction and two laps in the wrong direction. I also captured data recovering from left to center and right to center. More data was added using the image augmentaiton pipeline. The image augmentation pipeline creates a new augementated image for every 3rd image from the collected data and this is appened to the final data array.

#### **Image augmentation pipeline:**

My image augmentation pipeline consist of four steps:

##### **Image cropping:**
The image was cropped remove the unwanted to top and botton portion that would distract the model more. 

![alt text][image2]
|:--:| 
| *Original image* |

![alt text][image3]
|:--:| 
| *Cropped image* |

##### **Image resizing:**
The cropped image was further resized to half of the orignal size. This reduced the number of parameters and helped train the model fast.

##### **Image sheering:**
The above reized cropped image is further sheered.

![alt text][image4]
|:--:| 
| *Original image* |

![alt text][image5]
|:--:| 
| *Sheered image* |

##### **Image flipping:** 
Finally the image is flipped.

![alt text][image6]
|:--:| 
| *Original image* |

![alt text][image7]
|:--:| 
| *Flipped image* |


I finally randomly shuffled the data set.

I also used python generator instead of storing the data in the memory. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Below is video of the car driving autonomously using the trained model

[Video Output](./video_output.mp4)


