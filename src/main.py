import os
import pickle
from pprint import pprint

import numpy as np
import cv2

from resnet import resnet50
from loading_data import load_train_dataset
from testing import test
from evaluation import evaluate1, evaluate2
from training_svm import train_SVM

from const import TEST_DIR

# Don't train: load the saved models
TRAIN_FLAG = False

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
	train_features = load_train_dataset(model)

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
test_predictions, box_boundaries = test(edge_detection, model, svm_models)

# -- EVALUATION --

# Evaluation 1
evaluation1_results = evaluate1(test_predictions, class_labels)
pprint(evaluation1_results)

# Evaluation 2
evaluation2_results = evaluate2(test_predictions, box_boundaries, class_labels)
pprint(evaluation2_results)

# -- TESTING ON SAMPLE IMAGES --

idx = 0			# 0th image (for example)

'''
	Drawing the overlayed image with box boundaries

	:param idx - index of the image in test image dataset
	:param draw_all_boxes - flag for determining the boxes for overlayed image:
								- True: all 50 (max) boxes are drawn
								- False: only the 'best' predicted box is drawn
'''


def overlayed_image(idx, draw_all_boxes=False):
	# Reading image in index 'idx'
	image = cv2.imread(os.path.join(TEST_DIR, 'images', '{}.JPEG'.format(idx)))

	# Prediction data: label, boxes
	predicted_label, box_index = test_predictions[idx]

	if draw_all_boxes is True:
		for box in box_boundaries[idx]:
			box_x1, box_y1, box_x2, box_y2 = box
			cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 1)
	else:
		# Draw only the 'best' predicted box
		box_x1, box_y1, box_x2, box_y2 = box_boundaries[idx][box_index]
		cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 1)

	return image


# Displaying
cv2.imshow("test", overlayed_image(idx))
cv2.waitKey(0)
cv2.destroyAllWindows()