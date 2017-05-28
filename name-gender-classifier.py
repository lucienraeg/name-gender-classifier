# Lucien Rae 2017
# A classifier that predicts whether a name is male or female
# Uses a DecisionTree Classifier
# Data from a list of 258000 names, 50/50 ratio between male and female
# Features are the digits that correspond with the letters in the name
# Labels are the gender

import csv
import random
import numpy as np
source = 'names.csv'

# turns a word into a list of digits
def wordToDigits(word):
	wordD = [-1]*10 # give default values to the digits
	for i in range(min(len(list(word)),10)): # cut off the word if it's more than 10 characters
		char = list(word)[i]
		number = ord(char.lower()) - 97
		wordD[i] = (number)
	return(wordD)

print("Importing training data from '%s'" % (source))
# import training data
with open(source) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	names = []
	sexes = []
	for row in readCSV:
		name = row[1]
		sex = int(row[3] == 'boy') # 0 girl, 1 boy

		names.append(name)
		sexes.append(sex)

# gather and format features and labels
features = []
for name in names:
	# turn names into digits
	nameD = wordToDigits(name)
	features.append(nameD)
labels = sexes
genderName = {0: "Girl", 1: "Boy"}

# decide whether or not want to calculate accuracy (0 to go straight to manual testing)
calculate_accuracy = 0

print("Setting aside testing indicies")
# set aside a random selection of indicies for testing
test_idx = []
test_amount = 0.0004*calculate_accuracy # 1 = all

for i in range(int(len(names)*test_amount)):
	test_idx.append(random.randint(0,len(names)-1))

print("Sorting training data and omitting testing data")
# training data into numpy arrays, with testing data omitted
current_test_idx = 0
for i in test_idx:
	training_data = np.delete(features,test_idx, axis=0)
	training_target = np.delete(labels,test_idx)
	print("%i / %i: %i" % (current_test_idx,len(test_idx)-1,i))
	current_test_idx += 1

print("Sorting testing data")
# testing data into numpy arrays
testing_data = np.array(features)[test_idx]
testing_target = np.array(labels)[test_idx]

print("Defining Classifier")
# classifier
# from sklearn import tree
# clf = tree.DecisionTreeClassifier() # decide classifier

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

print("Fitting Classifier")
clf = clf.fit(features, labels) # find patterns in data


if calculate_accuracy:
	print("Testing Accuracy")
	# test accuracy
	test_length = len(testing_data)
	correct_predictions = 0

	for i in range(test_length):
		data = testing_data[i]
		target = testing_target[i]
		prediction = clf.predict([data])	
		if prediction == target:
			correct_predictions += 1

	print("Correct Predictions: %s / %s" % (correct_predictions,test_length))
	accuracy = (correct_predictions/test_length)*100
	print("Accuracy: %.2f" % (accuracy))

# manual testing
while True:
	namesToTest = input("What name/s would you like to test? ").split()
	for nameToTest in namesToTest:
		predictedGender = genderName[int(clf.predict([wordToDigits(nameToTest)]))]
		print("Predicted Gender of %s: %s" % (nameToTest.title(),predictedGender))