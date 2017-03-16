import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Convolution2D,MaxPooling2D,Activation,Dropout,Lambda,BatchNormalization
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint,EarlyStopping

from keras.preprocessing.image import random_shift

from scipy import ndimage
from scipy.misc import imresize
import cv2
import random


def nvidia_model(rows,cols,channes):
	model = Sequential()
	model.add(Lambda(lambda x:(x/255.0-0.5),input_shape=(rows,cols,channes)))
	model.add(BatchNormalization(mode=2,axis=3,input_shape=(rows,cols,channels)))
	model.add(Convolution2D(24,5,5,border_mode='same',subsample=(2,2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))

	model.add(Convolution2D(36,5,5,border_mode='same',subsample=(2,2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))

	model.add(Convolution2D(48,5,5,border_mode='same',subsample=(2,2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))

	model.add(Convolution2D(64,3,3,border_mode='same',subsample=(1,1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))

	model.add(Convolution2D(64,3,3,border_mode='same',subsample=(1,1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2,),strides=(1,1)))

	model.add(Flatten())
	model.add(Dense(128,activation='relu'))
	# model.add(Dropout(0.2))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(10, activation='relu'))
	# model.add(Dropout(0.2))
	model.add(Dense(1,activation='linear'))

	return model

# Use dataset provided by Udacity
path_prefix = './data/'
path = './data/driving_log.csv'

rows = 64
cols = 64
channels = 3

###-------------Data Augumentation-------START----------
def random_flip(img,angle):
	if random.random() > 0.5:
		img = cv2.flip(img,1)
		angle = -1.0 * angle
	return (img,angle)

def img_bright_change(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # convert it to hsv
	random_bright = .25 + np.random.uniform()
	hsv[:,:,2] = hsv[:,:,2] * random_bright

	img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	return img

# # random change lightness on one side of the image
def img_bright_change_2(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # convert it to hsv
	random_bright = random.randint(20,85)/100.0
	# hsv[:,:,2] = hsv[:,:,2] * random_bright
	h,w= image.shape[0],image.shape[1]
	if random.random() > 0.5:
		if random.random() > 0.5:
			hh = np.random.randint(int(h*0.6),int(h*0.8))
			hsv[:hh,:,2] = hsv[:hh,:,2] * random_bright
		else:
			hh = np.random.randint(int(h*0.5),int(h*0.8))
			hsv[hh:,:,2] = hsv[hh:,:,2] * random_bright
	else:
		if random.random() > 0.5:
			ww = np.random.randint(int(w*0.3),int(w*0.8))
			hsv[:,:ww,2] = hsv[:,:ww,2] * random_bright
		else:
			ww = np.random.randint(int(w*0.3),int(w*0.7))
			hsv[:,ww:,2] = hsv[:,ww:,2] * random_bright
	
	img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	return img

def img_bright_change_3(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # convert it to hsv
	random_bright = random.randint(25,85)/100.0
	# hsv[:,:,2] = hsv[:,:,2] * random_bright
	h,w= image.shape[0],image.shape[1]
	if random.random() > 0.5:
		hh = np.random.randint(int(h*0.5),int(h*0.9))
		hsv[:hh,:,2] = hsv[:hh,:,2] * random_bright
	else:
		hh = np.random.randint(int(h*0.5),int(h*0.9))
		hsv[hh:,:,2] = hsv[hh:,:,2] * random_bright

	img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	return img

def equalize_color_img(img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
	# equalize the histogram of the Y channel
	img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
	return  img_output

def random_adjust_gamma(image):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	gamma = np.random.uniform(0.5,1.5)
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def random_gamma_correction(img):
	img = img/255.0
	gamma = np.random.uniform(0.5,1.5)
	inv_gamma = 1.0/gamma
	img = cv2.pow(img, inv_gamma)
	return np.uint8(img*255)

def add_random_shadow(image):
	top_y = 320 * np.random.uniform()
	top_x = 0
	bot_x = 120
	bot_y = 320 * np.random.uniform()
	image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	shadow_mask = 0 * image_hls[:, :, 1]
	X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
	Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

	shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
	# random_bright = .25+.7*np.random.uniform()
	if np.random.randint(2) == 1:
		random_bright = .5
		cond1 = shadow_mask == 1
		cond0 = shadow_mask == 0
		if np.random.randint(2) == 1:
			image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
		else:
			image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
	image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

	return image

def random_shear(image, steering_angle, shear_range=200):
    """
    Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
	"""
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle

def crop(img,top_percent=0.35,bottom_percent=0.1):
	top = int(np.ceil(img.shape[0] * top_percent))
	bottom = img.shape[0] - int(np.ceil(img.shape[0] * bottom_percent))
	return img[top:bottom,:]
    

###--------------Data Augumentation-------END------------

###--------------data generator------START----------
def generate_imgs_1(img_list, label_list, batch_size=64):
	while True:
		X_batch = []
		y_batch = []

		for i in np.random.randint(0,len(img_list)-1,batch_size):
			img = ndimage.imread(img_list[i]).astype(np.float32)
			img = img[20:, ...]
			angle = label_list[i]
			img = imresize(img, (rows, cols, channels))

			if random.random() > 0.5:
				img = img_bright_change(img)

			if random.random() > 0.5:
				(img, angle) = img_flip(img, angle)

			img = add_random_shadow(img)

			X_batch.append(img)
			y_batch.append(angle)

		X = np.array(X_batch)
		Y = np.array(y_batch)

		yield (X, Y)


def img_process(img,angle):
	
	if random.random() < 0.2:
		img,angle = random_shear(img,angle)
	
	# if random.random() > 0.5:
	# 	img = add_random_shadow(img)	
	img = crop(img)
	img,angle = random_flip(img,angle)
	# img = random_gamma_correction(img)
	# img = img_bright_change_2(img) #this works
	if random.random() > 0.2:
		img = img_bright_change_3(img)

	img = imresize(img,(rows,cols))
	return img,angle	   
      
def generate_imgs_2(path, path_prefix, batch_size=64):
	data = pd.read_csv(path)
	nb_img = len(data)
	data.columns = ('Center','Left','Right','Angle','Throttle','Brake','Speed')
	while True:
		X_batch = []
		y_batch = []
		
		rnd_idx = np.random.randint(0,nb_img,batch_size)
		for i in rnd_idx:
			rnd_img = np.random.randint(0,3)
			if rnd_img == 0:
				img = data.iloc[i]['Left'].strip()
				angle = data.iloc[i]['Angle'] + 0.25
			if rnd_img == 1:
				img = data.iloc[i]['Center'].strip()
				angle = data.iloc[i]['Angle'] 
			if rnd_img == 2:
				img = data.iloc[i]['Right'].strip()
				angle = data.iloc[i]['Angle'] - 0.25
			img = cv2.imread(path_prefix+img)
			img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			new_img,new_angle = img_process(img,angle)
			X_batch.append(new_img)
			y_batch.append(new_angle)

		X = np.array(X_batch)
		Y = np.array(y_batch)
		
		yield (X, Y)


###--------------data generator------END------------

model = nvidia_model(rows,cols,channels)

lr = 1e-4
model.compile(optimizer=Adam(lr),loss='mse')
model.summary()


train_samples_per_epoch = 20032
val_samples_per_epoch = 6400
nb_epoch = 20

save_weights = ModelCheckpoint('./model/model_uda_3.h5',monitor='val_loss',save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss',patience=5,verbose=1,mode='auto')


model.fit_generator(generate_imgs_2(path,path_prefix),
					samples_per_epoch=train_samples_per_epoch,
					nb_epoch=nb_epoch,
                    validation_data=generate_imgs_2(path,path_prefix),
					nb_val_samples=val_samples_per_epoch,
					verbose=1,
                    callbacks=[save_weights,early_stopping])

# Not use validation dataset
# model.fit_generator(generate_img_2(X_train,y_train),samples_per_epoch=20000,nb_epoch=nb_epoch,verbose=1,callbacks=[save_weights])


# model.save_weights('.model/model_uda.h5', True)
#
# with open('.model/model_uda.json', 'w') as outfile:
# 	json.dump(model.to_json(), outfile)

# model.save('.model/model_uda_2.h5')












