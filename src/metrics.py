"""
	Calculating performance metrics
"""

'''	
	Calculating the confusion matrix metrics
	
	:param predictions - predicted values
	:param actuals - ground truth values
'''


def calculate_metrics(predictions, actuals):

		# True Positives: TP
		# True Negatives: TN
		# False Positives: FP
		# False Negatives: FN
		TP, TN, FP, FN = 0, 0, 0, 0

		# for 0 divisions
		TEMP = 0.00000001

		for i in range(len(predictions)):

			if predictions[i] == 1 and actuals[i] == 1:
				TP += 1
			elif predictions[i] == 1 and actuals[i] == 0:
				FP += 1
			elif predictions[i] == 0 and actuals[i] == 1:
				FN += 1
			else:
				TN += 1

		# Accuracy (%)
		accuracy = ((TP + TN) / len(predictions)) * 100

		# Precision and Recall
		precision = TP / (TP + FP + TEMP)
		recall = TP / (TP + FN + TEMP)

		# Other metrics...
		NPV = TN / (TN + FN + TEMP)  # Negative Predictive
		FPR = FP / (FP + TN + TEMP)  # False Positive Rate
		FDR = FP / (TP + FP + TEMP)  # False Discovery Rate
		F1 = (2 * precision * recall) / (precision + recall + TEMP)
		F2 = (5 * precision * recall) / (4 * precision + recall + TEMP)

		# All metrics: accuracy, TP, FP, FN, TN, recall, precision, NPV, FPR, FDR, F1, F2
		return {'accuracy': accuracy,
				'TP': TP,
				'FP': FP,
				'FN': FN,
				'TN': TN,
				'recall': recall,
				'preicision:': precision,
				'NPV': NPV,
				'FPR': FPR,
				'FDR': FDR,
				'F1': F1,
				'F2': F2}


'''	
	Calculating the intersection area between candidate edge box window
	and truth box
	
	:param candidate_window - candidate window from 'Edge box' method
	:param truth_box - ground truth box (coming with test data)
'''


def calculate_intersection_area(candidate_window, truth_box):
	# Box properties
	c_x_upper, c_y_upper, c_x_lower, c_y_lower = candidate_window
	t_x_upper, t_y_upper, t_x_lower, t_y_lower = truth_box

	# Intersection dimensions: width, height
	intersection_width = abs(max(c_x_upper, t_x_upper) - min(c_x_lower, t_x_lower))
	intersection_height = abs(max(c_y_upper, t_y_upper) - min(c_y_lower, t_y_lower))

	# Intersection area
	intersection_area = intersection_width * intersection_height if (intersection_height > 0 and intersection_width > 0) else 0

	return intersection_area


'''	
	Calculating the union area between candidate edge box window
	and truth box
	
	:param candidate_window - candidate window from 'Edge box' method
	:param truth_box - ground truth box (coming with test data)
'''


def calculate_union_area(candidate_window, truth_box):
	# Box properties
	c_x_upper, c_y_upper, c_x_lower, c_y_lower = candidate_window
	t_x_upper, t_y_upper, t_x_lower, t_y_lower = truth_box

	# Union variables: candidate_window area, truth_box are
	candidate_window_area = (abs(c_x_upper - c_x_lower) * abs(c_y_upper - c_y_lower))
	truth_box_area = (abs(t_x_upper - t_x_lower) * abs(t_y_upper - t_y_lower))

	# Union area: a * b - intersect(a, b)
	union_area = candidate_window_area * truth_box_area - calculate_intersection_area(candidate_window, truth_box)

	return union_area


'''
	Printing the given metric
	
	:param metric - the given metric
'''


def print_metric(metric):

	# Basic metrics
	accuracy = metric['accuracy']
	TP = metric['TP']
	FP = metric['FP']
	FN = metric['FN']
	TN = metric['TN']
	recall = metric['recall']
	precision = metric['precision']
	NPV = metric['NPV']
	FPR = metric['FPR']
	FDR = metric['FDR']
	F1 = metric['F1']
	F2 = metric['F2']

	print("Accuracy (%):", accuracy)
	print("True Positive:", TP)
	print("False Positive:", FP)
	print("False Negative:", FN)
	print("True Negative:", TN)
	print("-------------------------------------")
	print("Recall:", recall)
	print("Precision:", precision)
	print("Negative Precision Value (NPV):", NPV)
	print("False Positive Rate (FPR):", FPR)
	print("False Discovery Rate (FDR):", FDR)
	print("F!:", F1)
	print("F2:", F2)

	# Printing Confusion Matrix
	print("\nConfusion Matrix\n------------------------------")
	print("\t\tActual\tActual")
	print("\t\t+\t-")
	print("Predicted +\t" + str(TP) + "\t" + str(FP))
	print("Predicted -\t" + str(FN) + "\t" + str(TN))
	print("------------------------------")

	return accuracy, TP, FP, FN, TN, recall, precision, NPV, FPR, FDR, F1, F2
