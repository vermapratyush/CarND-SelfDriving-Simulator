**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (the model works best at 30mph)
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments atop of the cell.
There were a bunch of preprocessing that I removed, as the results were not satisfactory.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model is the one as mentioned in the [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
It involves 5 CNN layers (depth: 3 -> 24 -> 36 -> 48 -> 64 -> 64) with max pooling 

####2. Attempts to reduce overfitting in the model

The model contains dropouts in the last 2 CNN layers to avoid overfitting.
The loss in validation set and test set was also similar, which proves there is no over fitting. Later, I removed the test set, as it reduced the data over which I could train my network. And the test error didn't have enough significance, in this particular project.
The dataset also has flipped images to amortization the data.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Intially the data was skewed towards angle = 0.0 degree. Hence the # of such images had to be reduces by a factor.
The left and the right images were used to provide some randomness and durability in the dataset. The left image was added with a delta angle (0.3 radians) while the right image was subtracted by the same delta.

###Model Architecture and Training Strategy

####1. Solution Design Approach

I have used CNN for the purpose of solving the problem, as it provides an excellent solution in pattern recognition (image).
Layer of CNN on top of each other helps them learn complex patterns. Fully connected layer on top of the CNNs, provides the classification. The architecture of the model I have used the solution is inspired from the NVIDIA paper. The input size of the image to the Netural Net is different. 64x64 is what gave a good performance.

In order to find out how well the model fits the dataset, I split the dataset in train and validation set. Validation set provides the performance after each epoch. The error metrics used here is mean squared error, the value itself doesn't hold enough significance, the change in loss is important.

To combat the overfitting, I have added spatial dropout after the last 2 CNN layers. The loss on test set was similar to validation set, which validates that there is no overfitting. Later I removed the test set, inorder to get more data for training.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track at the first left turn after the bridge, and the following right turn. After a lot of troubleshooting and realised the distribution of the steering angle isn't good. For which I removed most of the pre-processing and was left with crop-resize-flip.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The model architecture involves 5 CNN and 4 FC layers. MaxPooling (padding=same) is done after every CNN.
Dropout is also added after 4th and 5th CNN.

Here is the model summary

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 64, 64, 24)    1824        convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 32, 32, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 32, 32, 36)    21636       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 16, 16, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 16, 16, 48)    43248       maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 8, 8, 48)      0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 8, 8, 64)      27712       maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 4, 4, 64)      0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
spatialdropout2d_1 (SpatialDropo (None, 4, 4, 64)      0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 4, 64)      36928       spatialdropout2d_1[0][0]         
____________________________________________________________________________________________________
maxpooling2d_5 (MaxPooling2D)    (None, 2, 2, 64)      0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
spatialdropout2d_2 (SpatialDropo (None, 2, 2, 64)      0           maxpooling2d_5[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 256)           0           spatialdropout2d_2[0][0]         
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           25700       flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
____________________________________________________________________________________________________
Total params: 162,619
Trainable params: 162,619
Non-trainable params: 0
____________________________________________________________________________________________________



####3. Creation of the Training Set & Training Process

I initally generated my own dataset by driving around the circuit for 4-5 laps. The steering\_angle distribution wasn\'t good enough, hence I falled back to the udacity dataset.

To augment the data sat, I use the left and the right images with delta added to the steering\_angle. I also randomly flipped the images to generate more dataset. This was done randomly so that model does not learn and overfit the augmented data.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

The ideal number of epochs was 5 as evidenced by the decrease in mean squared error. It was also possible to decrease the epoch to 4, however the loss remained constant which proves there is no over fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
