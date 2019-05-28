import numpy as np
import cv2

from resnet import resnet50
from loading_data import load_train_dataset
from testing import extract_candidate_windows, classify_localize, predict_features, test
from evaluation import read_test_data, evaluate1, evaluate2

from training_svm import train_SVM

# Creating ResNet50 model
model = resnet50(pretrained=True)

# Loading the train set (feature vectors and feature label)
train_features = load_train_dataset(model)

# Feature vectors and label for Train set
train_feature_vectors = [feature['feature_vec'] for feature in train_features]
train_feature_labels = [feature['label'] for feature in train_features]
class_labels = np.unique(train_feature_labels)

# -- TRAINING --

# One vs all training
svm_models = train_SVM(train_feature_vectors, train_feature_labels, class_labels)

# Edge detection object ('Edge box' method)
edge_detection = cv2.ximgproc.createStructuredEdgeDetection('../model/model.yml')

# -- TESTING --

# Testing results
test_predictions, box_boundaries = test(edge_detection, model, svm_models)

# -- EVALUATION --

# Evaluation 1
evaluation1_results = evaluate1(test_predictions, class_labels)

# Evaluation 2
evaluation2_results = evaluate2(test_predictions, box_boundaries, class_labels)
