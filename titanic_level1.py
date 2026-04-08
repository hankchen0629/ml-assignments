"""
File: titanic_level1.py
Name: 
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle website. This model is the most flexible among all
levels. You should do hyper-parameter tuning to find the best model.
"""

import math
from util import *
from collections import defaultdict
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating if it is training mode or testing mode
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	data = defaultdict(list)
	with open(filename, 'r') as f:
		if mode == 'Train':
			for line in f:
				if line[0].isdigit():
					line = line.strip()
					tokens = line.split(',')
					if tokens[6] != '' and tokens[12] != '':
						data['Survived'].append(int(tokens[1]))
						data['Pclass'].append(int(tokens[2]))
						if tokens[5] == 'male':
							data['Sex'].append(1)
						else:
							data['Sex'].append(0)
						data['Age'].append(float(tokens[6]))
						data['SibSp'].append(int(tokens[7]))
						data['Parch'].append(int(tokens[8]))
						data['Fare'].append(float(tokens[10]))
						if tokens[12] == 'S':
							data['Embarked'].append(0)
						elif tokens[12] == 'C':
							data['Embarked'].append(1)
						else:
							data['Embarked'].append(2)
		elif mode == 'Test':
			for line in f:
				if line[0].isdigit():
					line = line.strip()
					tokens = line.split(',')
					if tokens[1] != '':
						data['Pclass'].append(int(tokens[1]))
					else:
						data['Pclass'].append(round(sum(training_data['Pclass'])/len(training_data['Pclass']), 3))
					if tokens[4] == 'male':
						data['Sex'].append(1)
					elif tokens[4] == 'female':
						data['Sex'].append(0)
					else:
						data['Sex'].append(round(sum(training_data['Sex'])/len(training_data['Sex']), 3))
					if tokens[5] != '':
						data['Age'].append(float(tokens[5]))
					else:
						data['Age'].append(round(sum(training_data['Age']) / len(training_data['Age']), 3))
					if tokens[6] != '':
						data['SibSp'].append(int(tokens[6]))
					else:
						data['SibSp'].append(round(sum(training_data['SibSp']) / len(training_data['SibSp']), 3))
					if tokens[7] != '':
						data['Parch'].append(int(tokens[7]))
					else:
						data['Parch'].append(round(sum(training_data['Age']) / len(training_data['Parch']), 3))
					if tokens[9] != '':
						data['Fare'].append(float(tokens[9]))
					else:
						data['Fare'].append(round(sum(training_data['Fare']) / len(training_data['Fare']), 3))
					if tokens[11] == 'S':
						data['Embarked'].append(0)
					elif tokens[11] == 'C':
						data['Embarked'].append(1)
					elif tokens[11] == 'Q':
						data['Embarked'].append(2)
					else:
						data['Embarked'].append(round(sum(training_data['Embarked']) / len(training_data['Embarked']), 3))
	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	if feature == 'Sex':
		list_0 = []
		list_1 = []
		for i in range(len(data['Sex'])):
			if data['Sex'][i] == 0:
				list_0.append(1)
				list_1.append(0)
			else:
				list_0.append(0)
				list_1.append(1)
		data['Sex_0'] = list_0
		data['Sex_1'] = list_1
		del data['Sex']
	elif feature == 'Pclass':
		list_0 = []
		list_1 = []
		list_2 = []
		for i in range(len(data['Pclass'])):
			if data['Pclass'][i] == 1:
				list_0.append(1)
				list_1.append(0)
				list_2.append(0)
			elif data['Pclass'][i] == 2:
				list_0.append(0)
				list_1.append(1)
				list_2.append(0)
			else:
				list_0.append(0)
				list_1.append(0)
				list_2.append(1)
		data['Pclass_0'] = list_0
		data['Pclass_1'] = list_1
		data['Pclass_2'] = list_2
		del data['Pclass']
	elif feature == 'Embarked':
		list_0 = []
		list_1 = []
		list_2 = []
		for i in range(len(data['Embarked'])):
			if data['Embarked'][i] == 0:
				list_0.append(1)
				list_1.append(0)
				list_2.append(0)
			elif data['Embarked'][i] == 1:
				list_0.append(0)
				list_1.append(1)
				list_2.append(0)
			else:
				list_0.append(0)
				list_1.append(0)
				list_2.append(1)
		data['Embarked_0'] = list_0
		data['Embarked_1'] = list_1
		data['Embarked_2'] = list_2
		del data['Embarked']
	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	for key in data:
		normalized_list = []
		for i in range(len(data[key])):
			normalized_list.append((data[key][i] - min(data[key])) / (max(data[key]) - min(data[key])))
		data[key] = normalized_list
	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0
	# Step 2 : Start training
	for epoch in range(num_epochs):
		for i in range(len(labels)):
			# Step 3 : Feature Extract
			feature_vector = feature_extractor(inputs, i, degree)
			# Step 4 : Update weights
			h = sigmoid(dotProduct(weights, feature_vector))
			scale = -1 * alpha * (h - labels[i])
			increment(weights, scale, feature_vector)
	return weights


def feature_extractor(x: dict, num: int, degree: int):
	extracted_dict = {}
	keys = list(x.keys())
	for i in range(len(keys)):
		extracted_dict[keys[i]] = x[keys[i]][num]
	if degree == 2:
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				extracted_dict[keys[i] + keys[j]] = x[keys[i]][num] * x[keys[j]][num]
	return extracted_dict


def sigmoid(k):
	return 1/(1+math.exp(-k))
