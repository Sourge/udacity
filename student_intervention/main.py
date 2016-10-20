# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1]

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()

def preprocess_features(X):
	''' Preprocesses the student data and converts non-numeric binary variables into
		binary (0/1) variables. Converts categorical variables into dummy variables. '''

	# Initialize new output DataFrame
	output = pd.DataFrame(index=X.index)

	# Investigate each feature column for the data
	for col, col_data in X.iteritems():

		# If data type is non-numeric, replace all yes/no values with 1/0
		if col_data.dtype == object:
			col_data = col_data.replace(['yes', 'no'], [1, 0])

		# If data type is categorical, convert to dummy variables
		if col_data.dtype == object:
			# Example: 'school' => 'school_GP' and 'school_MS'
			col_data = pd.get_dummies(col_data, prefix=col)

		# Collect the revised columns
		output = output.join(col_data)

	return output

X_all = preprocess_features(X_all)

print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
index = 0
for column in X_all:
	data = X_all[column]
	student = []
	passing = []
	failed = []
	for i in range(0, len(data)):
		if y_all[i] == 'yes':
			passing.append(data[i])
		else:
			failed.append(data[i])
		student.append(data[i])

	# Stack the data
	avg_failed = sum(failed)/len(failed)
	avg_passing = sum(passing)/len(passing)
	avg_student = sum(student)/len(student)
	plt.figure()
	plt.xlabel(column+'')
	plt.ylabel('count')
	plt.title('Observation for ' + column)
	#plt.plot([0, 1, .5], [avg_failed, avg_passing, avg_student], 'ro')
	n, bins, patches = plt.hist([passing, failed], 30, stacked=True, normed=True)
	for patch in patches[0]:
		patch.set_facecolor('g')
	for patch in patches[1]:
		patch.set_facecolor('r')
	plt.show()
	print "\nFeature: ", column
	print "passed: ", avg_passing
	print "failed: ", avg_failed
