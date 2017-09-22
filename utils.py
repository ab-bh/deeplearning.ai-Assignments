# load necessary_train utilities
import os 
import numpy as np
from PIL import Image
import PIL
from scipy.misc import toimage


path1 = '/home/abhishek/Desktop/deeplearning/training'
path2 = '/home/abhishek/Desktop/deeplearning/testing'
def load_image():
	images_train = os.listdir(path1)
	images_test = os.listdir(path2)

	images_train_, y_train = [], []

	# load input data
	for image in images_train:
		im = Image.open(path1 + '/' + image)
		im = im.resize((64,64))
		im = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
		images_train_.append(im)
	images_train_ = np.asarray(images_train_)

	# load input labels
	for image in images_train:
		if image[:3]=='cat': y_train.append(1)
		else: y_train.append(0)
	y_train = np.asarray(y_train)
	y_train = y_train.reshape(1,y_train.shape[0])
	#toimage(images_train_[0]).show() # to see image back

	## testing data
	images_test_, y_test = [], []
	for image in images_test:
		im = Image.open(path2 + '/' + image)
		im = im.resize((64,64))
		im = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
		images_test_.append(im)
	images_test_ = np.asarray(images_test_)

	# load input labels
	for image in images_test:
		if image[:3]=='cat': y_test.append(1)
		else: y_test.append(0)
	y_test = np.asarray(y_test)
	y_test = y_test.reshape(1,y_test.shape[0])		
	
	# classes
	classes = np.asarray([b'not cat', b'cat'])
	return images_train_, y_train, images_test_, y_test, classes
load_image()