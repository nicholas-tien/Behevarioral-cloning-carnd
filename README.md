# Behevarioral-cloning-carnd
Train a model to drive car like a human.This project is from udacity self-driving car project.The aim is to train a model
that can drive a car like a human.It mainly use convolutional neural network to predict the steering angle.

##Denpencies
- Keras
- Opencv
- sklearn

##Data
We can use the dataset provided by udacity.Or we can collect our own dataset by using the simulator provided by udacity.
At first I collected my own dataset.But the result is not good.Maybe my driving skill is bad.So I use the datacity data.
Its result is good.

##Data aumentation
 The training dataset came from the track 1.The track 2 is much darker than track.And track 2 has a lot of shadow.
 To make the model more robust to environment impact and prevent overfitting,we have to do some data aumentation.
 For example,use image flip to balance the left and right turnig image,use hsv adjust or gamma corrction to change the lightness.


##Model
Here I use a nvidia-like model.[The nvidia paper is here](https://arxiv.org/abs/1604.07316).The model is as follows
![](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

##result

some driving capture images.

track 1 | track 2
---|---
<img src="https://github.com/nicholas-tien/Behevarioral-cloning-carnd/blob/master/image/track12.png?raw=true" width="40%" height="35%"> |<img src="https://github.com/nicholas-tien/Behevarioral-cloning-carnd/blob/master/image/track21.png?raw=true" width="40%" height="35%"> 
 <img src="https://github.com/nicholas-tien/Behevarioral-cloning-carnd/blob/master/image/track11.png?raw=true" width="40%" height="40%">|<img src="https://github.com/nicholas-tien/Behevarioral-cloning-carnd/blob/master/image/track22.png?raw=true" width="40%" height="35%"> 


