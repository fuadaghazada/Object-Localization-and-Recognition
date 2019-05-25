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
		precision = TP / (TP + FP)
		recall = TP / (TP + FN)

		# Other metrics...
		NPV = TN / (TN + FN)  # Negative Predictive
		FPR = FP / (FP + TN)  # False Positive Rate
		FDR = FP / (TP + FP)  # False Discovery Rate
		F1 = (2 * precision * recall) / (precision + recall)
		F2 = (5 * precision * recall) / (4 * precision + recall)

		# Basic metrics: accuracy, TP, FP, FN, TN
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

