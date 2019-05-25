"""
	Loading dataset
"""

import os

from const import TRAIN_DIR
from pre_processing import load_image, process_image, extract_feature_vector

'''
	Loading all train dataset feature  vectors and class labels
	
	:param model - ResNet model for extracting features
'''


def load_train_dataset(model):
	# Dataset features: [{feature_vec: ..., label: ...}, ...]
	features = []

	print("Pre-processing all the images in the train set...")

	for class_directory in os.listdir(TRAIN_DIR):
		if class_directory != '.DS_Store':
			for image_name in os.listdir(os.path.join(TRAIN_DIR, class_directory)):
				if image_name != '.DS_Store':

					# Loading the image
					image, _ = load_image(os.path.join(TRAIN_DIR, class_directory, image_name))
					print("\nLoaded image: '{}'...".format(image_name))

					# Pre-processing the image
					image = process_image(image)
					print("Pre-processed: '{}'...".format(image_name))

					# Extracting the feature vectors
					feature_vec = extract_feature_vector(image, model)
					print("Extracting feature vectors: '{}'...".format(image_name))

					# TODO: Normalize feature vector

					# Keeping the feature vectors and labels in list of dictionaries
					features.append({'feature_vec': feature_vec, 'label': class_directory})

	print("All images have been preprocessed and feature vectors are extracted!")

	return features



