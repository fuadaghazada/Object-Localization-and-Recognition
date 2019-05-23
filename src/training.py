"""
	Training steps
"""

import os
from sklearn.svm import SVC

from const import TRAIN_DIR
from pre_processing import process_image, extract_feature_vector
from net.resnet import resnet50

'''
	Loading all train dataset feature  vectors and class labels
'''


def load_dataset(model):
	# Dataset features: [{feature_vec: ..., label: ...}, ...]
	features = []

	for class_directory in os.listdir(TRAIN_DIR):
		if class_directory != '.DS_Store':
			for image_name in os.listdir(os.path.join(TRAIN_DIR, class_directory)):
				if image_name != '.DS_Store':
					# Pre-processing the image
					image = process_image(TRAIN_DIR, os.path.join(class_directory, image_name))

					# Extracting the feature vectors
					feature_vec = extract_feature_vector(image, model)

					# TODO: Normalize feature vector

					# Keeping the feature vectors and labels in list of dictionaries
					features.append({'feature_vec': feature_vec, 'label': class_directory})

	return features


'''
	Training
'''


def train():
	model = resnet50(pretrained=True)

	# Feature vectors and labels
	features = load_dataset(model)
	feature_vecs = [feature['feature_vec'] for feature in features]
	labels = [feature['label'] for feature in features]


train()




