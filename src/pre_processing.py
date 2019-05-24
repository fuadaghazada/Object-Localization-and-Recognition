"""
	Pre-processing steps
"""

import os

import torch
import cv2
from PIL import Image
import numpy as np

'''
	=================
	| NORMALIZATION |
	=================
'''

'''
	Loads an image from the given path to the root directory
	
	:param str path - path for the train or test images in dataset
'''


def load_image(path):
	# Loading the image
	image = Image.open(path).convert('RGB')

	# Converting to the numpy as np
	image_arr = np.asarray(image)

	return image, image_arr


'''
	Processing the given image matrix
		- Padding 
		- Rescaling 
		- Normalization
		
	:return np result_image - the image after processing
'''


def process_image(image):
	# -- PADDING --
	result_image, result_image_arr = add_padding(image)

	# -- RESCALING --
	result_image = rescale(result_image_arr, (224, 224))

	# -- NORMALIZATION --
	result_image = normalize(result_image)

	return np.float32(result_image)


'''
	Pad the image with minimum number of zeros 
	so that it becomes a square image
	
	https://pillow.readthedocs.io/en/4.1.x/reference/Image.html
	
	:param image - the given image 
	:return np result_image - the image after adding padding
'''


def add_padding(image):
	# Image dimensions
	im_height, im_width, _ = np.asarray(image).shape

	# Determining the larger side
	new_size = max(im_height, im_width)

	# Creating the black black background image
	result_image = Image.new('RGB', (new_size, new_size))

	# Coordinates for 'pasting': left, upper, right, lower
	center_coord = ((new_size - im_width) // 2, (new_size - im_height) // 2,
					im_width + (new_size - im_width) // 2, im_height + (new_size - im_height) // 2)

	# 'Pasting' the image to the center of the black background
	result_image.paste(image, box=center_coord)

	return result_image, np.asarray(result_image)


'''
	Rescaling the given image to the given dimensions
	
	https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
	
	:param np image - the given image matrix
	:param tuple dimension - the desired dimension for rescaling
'''


def rescale(image, dimension):
	return cv2.resize(image, dsize=dimension)


'''
	Normalizing the given image
	
	:param np image - the given image
	:return np result_image - the image after adding padding
'''


def normalize(image):
	# Divide an image by 255 so that its values are in the range [0, 1]
	result_image = np.float32(image / 255)

	# Subtract 0.485, 0.456, 0.406 from red, green and blue channels, respectively
	result_image -= np.array([0.485, 0.456, 0.406])

	# Divide red, green and blue channels by 0.229, 0.224, 0.225, respectively
	result_image /= np.array([0.229, 0.224, 0.225])

	return np.asarray(result_image)


'''
	======================
	| FEATURE EXTRACTION |
	======================
'''

'''
	Extracting 2048-dimensional vectors for the training models
	
	:param np image - the given image
'''


def extract_feature_vector(image, model):
	# Append an augmented dimension to indicate batch_size, which is one
	result_image = np.reshape(image, [1, 224, 224, 3])

	# model takes as input images of size [batch_size, 3, im_height, im_width]
	result_image = np.transpose(result_image, [0, 3, 1, 2])

	# Convert the Numpy image to torch.FloatTensor
	result_image = torch.from_numpy(result_image)

	# Extract features
	feature_vector = model(result_image)

	# convert the features of type torch.FloatTensor to a Numpy array
	feature_vector = feature_vector.detach().numpy()

	return feature_vector
