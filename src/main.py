import pickle
from pprint import pprint

import numpy as np
import cv2

from resnet import resnet50
from loading_data import load_train_dataset
from testing import extract_candidate_windows, classify_localize, predict_features, test
from evaluation import read_test_data, evaluate1, evaluate2

from training_svm import train_SVM

# Don't train: load the saved models
TRAIN_FLAG = False

class_labels = np.asarray(['n01615121', 'n02099601', 'n02123159', 'n02129604', 'n02317335',
						   'n02391049', 'n02410509', 'n02422699', 'n02481823', 'n02504458'])

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