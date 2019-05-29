"""
	Testing (Part 5)
"""

import os
import cv2
import numpy as np
from PIL import Image

from const import TEST_DIR
from pre_processing import process_image, extract_feature_vector, apply_l2_normalization

'''
	Testing
	
	:param edge_detection - edge_detection object generated from model for edge detection
	:param model - ResNet model for extracting features
	:param classifiers - one-vs-all classifiers for predicting
'''


def test(edge_detection, model, classifiers, l2_normalization):

	label_predictions = []
	boundary_edges = []

	for i in range(100):
		# Name of the image file
		image_name = "{}.JPEG".format(i)

		if image_name != '.DS_Store':
			print("\nTesting: '{}'...".format(image_name))

			image_path = os.path.join(TEST_DIR, 'images', image_name)

			# Extracting the candidate windows
			boxes = extract_candidate_windows(image_path, edge_detection)
			boundary_edges.append(boxes)
			print("Extracting candidate windows: '{}'...".format(image_name))

			# Classifying and Localizing
			test_features = classify_localize(image_path, boxes, model, l2_normalization)
			print("Classifying and Localizing: '{}'...".format(image_name))

			# Predicting
			predictions = predict_features(test_features, classifiers)
			print("Predictions: '{}'...".format(image_name))

			# Choosing the best prediction
			best_prediction = np.unravel_index(np.argmax(predictions, axis=None), predictions.shape)
			label_predictions.append(best_prediction)

	# Converting to numpy array
	label_predictions = np.asarray(label_predictions)
	boundary_edges = np.asarray(boundary_edges)

	return label_predictions, boundary_edges


'''
	Extracting candidate windows 
	
	Edge boxes:
		https://github.com/opencv/opencv_contrib/blob/96ea9a0d8a2dee4ec97ebdec8f79f3c8a24de3b0/modules/ximgproc/samples/edgeboxes_demo.py
	
	:param image_path - path of the image 
	:param edge_detection - edge_detection object generated from model for edge detection
'''


def extract_candidate_windows(image_path, edge_detection):

	# Reading the image and converting it to RGB
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# 'Edge Boxes' method from OpenCV source
	edges = edge_detection.detectEdges(np.float32(image) / 255.0)
	orimap = edge_detection.computeOrientation(edges)
	edges = edge_detection.edgesNms(edges, orimap)

	edge_boxes = cv2.ximgproc.createEdgeBoxes()
	edge_boxes.setMaxBoxes(50)
	boxes = edge_boxes.getBoundingBoxes(edges, orimap)

	return boxes


'''
	Classifying and localizing the extracted candidate windows 
	
	:param image_path - path of the image
	:param boxes - candidate windows extracted for the given image 
	:param model - ResNet model for extracting features
'''


def classify_localize(image_path, boxes, model, l2_normalization):

	test_features = []

	# Reading the Image as PIL image (for crop)
	image = Image.open(image_path).convert('RGB')

	for box in boxes:
		b_x, b_y, b_w, b_h = box

		# Cropping the edge box from the original test image
		cropped = image.crop((b_x, b_y, (b_x + b_w), (b_y + b_h)))

		# Pre-process the cropped image
		cropped = process_image(cropped)

		# Extracting feature vectors
		test_feature_vec = extract_feature_vector(cropped, model)

		# Apply l2 normalization on the feature vector
		# (optional, specified by the parameter "l2_normalization")
		if l2_normalization:
			test_feature_vec = apply_l2_normalization(test_feature_vec)

		# Keeping the extracted features in the test features list
		test_features.append(test_feature_vec)

	test_features = np.asarray(test_features)

	return np.asarray(test_features)


'''
	Predicting the given test features with the given classifiers
	
	:param test_features - the list of test feature vectors 
	:param classifiers - one-vs-all classifiers for predicting 
'''


def predict_features(test_features, classifiers):

	# List of predictions for each classifier
	predictions = []

	# Reshaping
	test_features = np.asarray(test_features).reshape(-1, 2048)

	for classifier in classifiers:
		prediction = classifier.predict_proba(test_features)
		predictions.append(prediction[:, 1])

	predictions = np.asarray(predictions)

	return predictions

