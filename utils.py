# load necessary_train utilities
import os 
import numpy as np
from PIL import Image
import PIL
from scipy.misc import toimage

path1 ='/home/abhishek/Desktop/Scratch_NN/training'
path2 ='/home/abhishek/Desktop/Scratch_NN/testing'

def image_to_arr(image_list, path):
	images_list_ = []
	for image in image_list:
		im = Image.open(path + '/' + image)
		im = im.resize((64,64))
		im = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
		images_list_.append(im)
	images_list_ = np.asarray(images_list_)
	images_list_ = images_list_.astype('float32')
	return images_list_

def gen_labels(image_list, path):
	y = []
	for image in image_list:
		if image[:3]=='cat': y.append(1)
		else: y.append(0)
	y = np.asarray(y)
	y = y.astype('float32')
	return y.reshape(1,y.shape[0])

def load_image():

	# load training/testing data
	images_train = os.listdir(path1)
	images_test = os.listdir(path2)

	# train image data and labels will be stored here
	images_train_ = image_to_arr(images_train, path1)

	## test image data will be stored here 
	images_test_  = image_to_arr(images_test, path2)
	
	#toimage(images_train_[0]).show() # to see image back

	# load train/test labels
	y_train = gen_labels(images_train, path1)
	y_test  = gen_labels(images_test, path2) 
	
	return images_train_, y_train, images_test_, y_test
