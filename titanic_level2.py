"""
File: titanic_level2.py
Name: 
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle website. Hyper-parameters tuning are not required due to its
high level of abstraction, which makes it easier to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'
nan_cache = {}


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	data.pop('PassengerId')
	data.pop('Name')
	data.pop('Ticket')
	data.pop('Cabin')
	if mode == 'Train':
		data = data.dropna(subset=['Age', 'Embarked'])
		labels = data.pop('Survived')
		data.loc[data.Sex == 'male', 'Sex'] = 1
		data.loc[data.Sex == 'female', 'Sex'] = 0
		data.Embarked = data.Embarked.replace({'S': 0, 'C': 1, 'Q': 2})
		nan_cache['Age'] = round(data['Age'].mean(), 3)
		nan_cache['Fare'] = round(data['Fare'].mean(), 3)
		return data, labels
	elif mode == 'Test':
		data.loc[data.Sex == 'male', 'Sex'] = 1
		data.loc[data.Sex == 'female', 'Sex'] = 0
		data.Embarked = data.Embarked.replace({'S': 0, 'C': 1, 'Q': 2})
		data['Age'].fillna(nan_cache['Age'], inplace=True)
		data['Fare'].fillna(nan_cache['Fare'], inplace=True)
		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	data = pd.get_dummies(data, columns=[feature])
	if feature == 'Pclass':
		data = data.rename(columns={'Pclass_1': 'Pclass_0', 'Pclass_2': 'Pclass_1', 'Pclass_3': 'Pclass_2'})
	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	if mode == 'Train':
		standardizer = preprocessing.StandardScaler()
		data = standardizer.fit_transform(data)
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy on degree1;
	~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimal places)
	TODO: real accuracy on degree1 -> 0.80196629
	TODO: real accuracy on degree2 -> 0.83707865
	TODO: real accuracy on degree3 -> 0.87640449
	"""
	data, y = data_preprocess(TRAIN_FILE, mode='Train')
	x_train = one_hot_encoding(data, 'Sex')
	x_train = one_hot_encoding(x_train, 'Pclass')
	x_train = one_hot_encoding(x_train, 'Embarked')
	standardizer = preprocessing.StandardScaler()
	x_train = standardizer.fit_transform(x_train)

	poly_fea_extractor = preprocessing.PolynomialFeatures(degree=3)
	x_train = poly_fea_extractor.fit_transform(x_train)

	# Training
	h = linear_model.LogisticRegression(max_iter=10000)
	predictor = h.fit(x_train, y)
	acc = predictor.score(x_train, y)
	print('Degree 1 Training Acc:', acc)


if __name__ == '__main__':
	main()
