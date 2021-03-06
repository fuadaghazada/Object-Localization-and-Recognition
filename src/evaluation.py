"""
	Evaluation checks
"""

import os
import sys
import csv

import numpy as np

from const import TEST_DIR
from metrics import calculate_metrics, calculate_intersection_area, calculate_union_area

# Actual names
class_names = np.asarray(['eagle', 'dog', 'cat', 'tiger', 'star',
						  'zebra', 'bison', 'antelope', 'chimpanzee', 'elephant'])

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
		predictions[predictions != i] = -1
		predictions[predictions == i] = 1
		predictions[predictions == -1] = 0

		# Ground truth
		ground_truth = test_actual_labels.copy()
		ground_truth[ground_truth != i] = -1
		ground_truth[ground_truth == i] = 1
		ground_truth[ground_truth == -1] = 0

		# Calculating confusion matrix metrics
		metrics = calculate_metrics(predictions=predictions, actuals=ground_truth)
		evaluation_results.append({'label': i, 'metrics': metrics})

	return evaluation_results


'''
	Evaluate2: Localization accuracy
		- Localization accuracy: intersection / area
		- Overall accuracy in terms of percentage
		
	:param test_predictions - result labels after predicting on test images
	:param boundary_boxes - result boxes after predicting on test images
	:param class_labels - unique class labels of the dataset  	
'''


def evaluate2(test_predictions, boundary_boxes, class_labels):

	# Writing the results to csv file
	writer = csv.writer(open('../results.csv', 'w'))
	writer.writerow(['Test data', 'Actual label', 'Predicted label', 'Location accuracy'])

	# For calculating overall accuracy
	num_correct_classes, num_correct_locs = 0, 0
	total = len(test_predictions)

	# Localization accuracies for test images
	localization_accuracies = []

	# Ground truth labels and boxes
	test_data = read_test_data(class_labels)
	test_actual_labels = test_data['labels']
	test_actual_boxes = test_data['boxes']

	for i, test_prediction in enumerate(test_predictions):
		prediction_label = test_prediction[0]
		prediction_box_index = test_prediction[1]

		# Predicted candidate window
		candidate_window = boundary_boxes[i, prediction_box_index]
		c_x, c_y, c_w, c_h = candidate_window

		# Localization accuracy: intersection / union
		intersect_area = calculate_intersection_area([c_x, c_y, c_x + c_w, c_y + c_h], test_actual_boxes[i])
		union_area = calculate_union_area([c_x, c_y, c_x + c_w, c_y + c_h], test_actual_boxes[i])

		# localization accuracy (in percentage notation, out of 100)
		percentage = float(intersect_area / union_area) * 100
		localization_accuracies.append(percentage)

		# Counting overall correct predictions
		if prediction_label == test_actual_labels[i]:
			num_correct_classes += 1

		if  percentage >= 50:
			num_correct_locs += 1

		# Writing row
		writer.writerow(['{}'.format(i + 1), class_names[test_actual_labels[i]], class_names[prediction_label], str(round(percentage, 2))])

	localization_accuracies = np.array(localization_accuracies)

	return {'overall_classification_accuracy': float(num_correct_classes / total),
			'overall_localization_accuracy': float(num_correct_locs / total),
			'localization_accuracies': localization_accuracies}

