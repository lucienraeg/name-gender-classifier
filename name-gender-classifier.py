# Lucien Rae 2017
# A classifier that predicts whether a name is male or female
# Uses a DecisionTree Classifier
# Data from a list of 258000 names, 50/50 ratio between male and female
# Features are the digits that correspond with the letters in the name
# Labels are the gender

import csv
import random
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
source = 'names.csv'

# turns a word into a list of digits
def wordToDigits(word):
	wordD = [-1]*10 # give default values to the digits
	for i in range(min(len(list(word)),10)): # cut off the word if it's more than 10 characters
		char = list(word)[i]
		number = ord(char.lower()) - 97
		wordD[i] = (number)
	return(wordD)

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
X = []
for name in names:
	# turn names into digits
	nameD = wordToDigits(name)
	X.append(nameD)
y = sexes
genderName = {0: "Girl", 1: "Boy"}

# split training / testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .0002)

# classifier
clf = tree.DecisionTreeClassifier() # decide classifier

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier()

# fit to classifier
clf = clf.fit(X_train, y_train) # find patterns in data

# test accuracy
predictions = clf.predict(X_test)
print("Accuracy: %.2f" % accuracy_score(y_test, predictions))

# manual testing
while True:
	namesToTest = input("What name/s would you like to test? ").split()
	for nameToTest in namesToTest:
		predictedGender = genderName[int(clf.predict([wordToDigits(nameToTest)]))]
		print("Predicted Gender of %s: %s" % (nameToTest.title(),predictedGender))
