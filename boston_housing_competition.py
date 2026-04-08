"""
File: boston_housing_competition.py
Name: 
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, linear_model
from sklearn import ensemble
from sklearn import metrics
from xgboost import XGBRegressor

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'


def main():
	train_data = pd.read_csv(TRAIN_FILE)
	# print(train_data.count())
	test_data = pd.read_csv(TEST_FILE)
	# print(test_data.count())

	y = train_data.pop('medv')
	features = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'lstat']
	x_train = train_data[features]
	x_train, x_val, y_train, y_val = train_test_split(x_train, y,  test_size=0.5, random_state=11)
	xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=10, subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=10, gamma=1, objective='reg:squarederror', random_state=13, n_jobs=-1)
	predictor = xgb.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="rmse", early_stopping_rounds=20, verbose=False)
	acc = predictor.score(x_train, y_train)
	print('Train Acc:', acc)
	prediction_val = xgb.predict(x_val)
	mean_square_error = metrics.mean_squared_error(prediction_val, y_val) ** 0.5
	print('Mean Square Error:', mean_square_error)

	x_test = test_data[features]
	x_test_id = test_data['ID']
	prediction_test = predictor.predict(x_test)
	out_file(prediction_test, 'boston_housing.csv', x_test_id)


def out_file(predictions, filename, id):
	print('\n===============================================')
	print(f'Writing predictions to --> {filename}')
	with open(filename, 'w') as out:
		out.write('ID,medv\n')
		for i in range(len(id)):
			out.write(str(id[i]) + ',' + str(predictions[i]) + '\n')
	print('===============================================')


if __name__ == '__main__':
	main()
