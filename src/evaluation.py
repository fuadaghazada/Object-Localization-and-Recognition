"""
	Evaluation checks
"""

import os
import sys

import numpy as np

from const import TEST_DIR
from metrics import calculate_metrics


'''
	Reading the test labels
	
	:param class_labels - unique class labels of the dataset
'''


def read_test_data(class_labels):

	# Test data
	test_labels = []
	test_boxes = []

	try:
		# Opening the file
		file = open(os.path.join(TEST_DIR, 'bounding_box.txt'))

		# Reading the file
		for line in file:
			temp = line.strip().replace(' ',  '').split(',')
			test_labels.append(temp[0])
			test_boxes.append(temp[1:])

	except Exception as e:
		raise(e)
		sys.exit()

	test_labels = np.asarray(test_labels)
	test_boxes = np.asarray(test_boxes, dtype=np.int)

	# Converting string labels to integer indices
	for i, label in enumerate(class_labels):
		test_labels[test_labels == label] = i
	test_labels = np.asarray(test_labels, dtype=np.int)

	return {'labels': test_labels,
			'boxes': test_boxes}


'''
	Evaluate1: Classification accuracy
		- Calculating confusion matrix: precision, recall
		- Overall accuracy in terms of percentage
	
	:param test_predictions - results after predicting on test images
	:param class_labels - unique class labels of the dataset  
'''


def evaluate1(test_predictions, class_labels):

	# Results of evaluation 1
	evaluation_results = []

	# Ground truth labels
	test_actual_labels = read_test_data(class_labels)['labels']

	for i in range(len(class_labels)):
		# Test predictions
		predictions = test_predictions[:, 0].copy()
		predictions[predictions == i] = 1
		predictions[predictions != i] = 0
		predictions[predictions == -1] = -1

		# Ground truth
		ground_truth = test_actual_labels.copy()
		ground_truth[ground_truth == i] = 1
		ground_truth[ground_truth != i] = 0
		ground_truth[ground_truth == -1] = -1

		# Calculating confusion matrix metrics
		metrics = calculate_metrics(predictions=predictions, actuals=ground_truth)
		evaluation_results.append({'label': 1, 'metrics': metrics})

	return evaluation_results


'''
	Evaluate2: Localization accuracy
'''


def evaluate2():
	# TODO: Implement...
	pass
