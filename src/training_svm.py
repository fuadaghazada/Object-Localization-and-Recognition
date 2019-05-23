"""
	Training SVM (one vs all)
"""

import numpy as np
from sklearn.svm import SVC

'''
	Training SVM models for each object type

	:param train_feature_vectors - feature vectors of train images
	:param train_feature_labels - feature labels of train images
	:param class_labels - unique class labels in train dataset
'''


def train_SVM(train_feature_vectors, train_feature_labels, class_labals):
	# SVM model list
	svm_s = []

	# Train vectors in numpy array [# of training, 2048]
	train_vectors = np.asarray(train_feature_vectors).reshape(-1, 2048)

	for label in class_labals:
		# Choosing object type 'label' setting the labels to 1: for the other object type to 0
		train_labels = np.asarray(train_feature_labels)
		train_labels[train_labels != label] = 0
		train_labels[train_labels == label] = 1

		# Training step
		svm = SVC(kernel='rbf', C=0.1, gamma=1 / 8)
		svm.fit(train_vectors, train_labels)

		# Keeping the trained model for the object type 'label'
		svm_s.append(svm)

	return svm_s
