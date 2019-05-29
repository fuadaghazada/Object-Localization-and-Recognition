import os
import pickle
from pprint import pprint

import numpy as np
import cv2
import matplotlib.pyplot as plt

from resnet import resnet50
from loading_data import load_train_dataset
from testing import test
from evaluation import evaluate1, evaluate2, read_test_data
from training_svm import train_SVM

from const import TEST_DIR

# Don't train: load the saved models
TRAIN_FLAG = True

# Specify whether to apply L2-normalization on feature vectors
NORMALIZATION_FLAG = False

# Class label (unique)
class_labels = np.asarray(['n01615121', 'n02099601', 'n02123159', 'n02129604', 'n02317335',
						   'n02391049', 'n02410509', 'n02422699', 'n02481823', 'n02504458'])

# Actual names
class_names = np.asarray(['eagle', 'dog', 'cat', 'tiger', 'star',
						  'zebra', 'bison', 'antelope', 'chimpanzee', 'elephant'])

# Creating ResNet50 model
model = resnet50(pretrained=True)

if TRAIN_FLAG is True:
	# Loading the train set (feature vectors and feature label)
	train_features = load_train_dataset(model, NORMALIZATION_FLAG)

	# Feature vectors and label for Train set
	train_feature_vectors = [feature['feature_vec'] for feature in train_features]
	train_feature_labels = [feature['label'] for feature in train_features]

	# -- TRAINING --

	# One vs all training
	svm_models = train_SVM(train_feature_vectors, train_feature_labels, class_labels)

	# Saving the models
	for i, svm_model in enumerate(svm_models):
		pickle.dump(svm_model, open('../model/svm_model_{}.obj'.format(i), 'wb'))

# Edge detection object ('Edge box' method)
edge_detection = cv2.ximgproc.createStructuredEdgeDetection('../model/model.yml')

# -- TESTING --

# Loading the models
svm_models = []
for i in range(10):
	svm_model = pickle.load(open('../model/svm_model_{}.obj'.format(i), 'rb'))
	svm_models.append(svm_model)

# Testing results
test_predictions, box_boundaries = test(edge_detection, model, svm_models, NORMALIZATION_FLAG)

# -- EVALUATION --

# Evaluation 1
evaluation1_results = evaluate1(test_predictions, class_labels)
pprint(evaluation1_results)

# Evaluation 2
evaluation2_results = evaluate2(test_predictions, box_boundaries, class_labels)
pprint(evaluation2_results)

# -- TESTING ON SAMPLE IMAGES --

'''
	Drawing the overlayed image with box boundaries
	
	:param idx - index of the image in test image dataset
	:param draw_all_boxes - flag for determining the boxes for overlayed image:
								- True: all 50 (max) boxes are drawn
								- False: only the 'best' predicted box is drawn
	:param draw_labels - flag for determining putting the labels near the boxes
'''


def overlayed_image(idx, draw_all_boxes=False, draw_labels=False):
	# Reading image in index 'idx'
	image = cv2.imread(os.path.join(TEST_DIR, 'images', '{}.JPEG'.format(idx)))

	# Prediction data: label, boxes
	predicted_label, box_index = test_predictions[idx]

	# Ground truth data
	ground_truth_data = read_test_data(class_labels)
	ground_truth_labels = ground_truth_data['labels']
	ground_truth_boxes = ground_truth_data['boxes']

	# Drawing ground true box
	true_label = ground_truth_labels[idx]
	box_x1, box_y1, box_x2, box_y2 = ground_truth_boxes[idx]
	cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)

	print("Predicted label:", class_names[predicted_label])
	print("Ground truth label:", class_names[true_label])
	print("Localization accuracy:", evaluation2_results['localization_accuracies'][idx])

	# Ground Truth label text
	if draw_labels is True:
		cv2.putText(image, class_names[true_label], (box_x1, box_y2 + 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0))

	# Drawing
	if draw_all_boxes is True:
		for box in box_boundaries[idx]:
			box_x1, box_y1, width, height = box
			box_x2, box_y2 = box_x1 + width, box_y1 + height
			cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 2)
	else:
		# Draw only the 'best' predicted box
		box_x1, box_y1, width, height = box_boundaries[idx][box_index]
		box_x2, box_y2 = box_x1 + width, box_y1 + height
		cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 2)

	# Prediction label text
	if draw_labels is True:
		cv2.putText(image, class_names[true_label], (box_x1, box_y1 - 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 0, 0))

	return image


# Displaying
plt.imshow(overlayed_image(59))
plt.show()