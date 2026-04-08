"""
File: validEmailAddress_2.py
Name:
----------------------------
Please construct your own feature vectors
and try to surpass the accuracy achieved by
Jerry's feature vector in validEmailAddress.py.
feature1:  TODO: NO '@'
feature2:  TODO: NO '.'
feature3:  TODO: '.' after the last'@'
feature4:  TODO: The string before the last '@' is alpha or digit
feature5:  TODO: The strings after the last '@' is alpha
feature6:  TODO: Consecutive '.' while not in quote, EX
feature7:  TODO: Start with '.'
feature8:  TODO: no single string in quote
feature9:  TODO: '@' at start and finish
feature10: TODO: '(),:;<>[\]' not in quote

Accuracy of your model: TODO:
"""
import numpy as np

WEIGHT = [                           # The weight vector selected by you
	[-10],                              # (Please fill in your own weights)
	[-10],
	[0.6],
	[0.6],
	[0.6],
	[-10],
	[-10],
	[-10],
	[-10],
	[-10]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	maybe_email_list = read_in_data()
	count = 1
	correct = 0
	weight_vector = np.array(WEIGHT)
	weight_vector = weight_vector.T
	for maybe_email in maybe_email_list:
		feature_vector = feature_extractor(maybe_email)
		print(feature_vector)
		points = weight_vector.dot(feature_vector)
		print(points)
		if count <= 13:
			if points < 0:
				correct += 1
		else:
			if points > 0:
				correct += 1
		count += 1
	print('Accuracy:', correct / 26)


def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with value 0's and 1's
	"""
	feature_vector = np.array([0] * len(WEIGHT))
	for i in range(len(feature_vector)):
		if i == 0:
			feature_vector[i] = 0 if '@' in maybe_email else 1
		elif i == 1:
			feature_vector[i] = 0 if '.' in maybe_email else 1
		elif i == 2:
			if not feature_vector[0] and not feature_vector[1]:
				if '.' in maybe_email.split('@')[-1]:
					feature_vector[i] = 1
				else:
					feature_vector[i] = 0
		elif i == 3:
			if not feature_vector[0] and maybe_email.split('@')[-2]:
				ele = maybe_email.split('@')[-2][-1]
				if ele.isalpha() or ele.isdigit():
					feature_vector[i] = 1
				else:
					feature_vector[i] = 0
		elif i == 4:
			if not feature_vector[0] and maybe_email.split('@')[-1]:
				if maybe_email.split('@')[-1][0].isalpha():
					feature_vector[i] = 1
				else:
					feature_vector[i] = 0
		elif i == 5:
			if not feature_vector[1]:
				switch = 0
				for j in range(len(maybe_email)):
					if maybe_email[j] == '\"':
						switch += 1
					elif maybe_email[j] == '.' and j != 0:
						if maybe_email[j-1] == '.' and switch % 2 == 0:
							feature_vector[i] = 1
		elif i == 6:
			if maybe_email[0] == '.':
				feature_vector[i] = 1
		elif i == 7:
			switch = 0
			check_alpha = ''
			for j in range(len(maybe_email)):
				if maybe_email[j] == '\"':
					switch += 1
				elif switch % 2 == 1:
					check_alpha += maybe_email[j]
				elif switch % 2 == 0:
					if check_alpha.isalpha():
						feature_vector[i] = 1
					check_alpha = ''
		elif i == 8:
			if maybe_email[0] is '@' or maybe_email[-1] is '@':
				feature_vector[i] = 1
		elif i == 9:
			switch = 0
			lst = ['(', ')', ',', ':', ';', '<', '>', '[', ']', '\\']
			for j in range(len(maybe_email)):
				if maybe_email[j] == '\"':
					switch += 1
				elif switch % 2 == 0:
					if maybe_email[j] in lst:
						feature_vector[i] = 1
						break
	print(maybe_email)
	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that may be valid email addresses
	"""
	emails = []
	with open(DATA_FILE, 'r') as f:
		for line in f:
			emails.append(line.strip())
	return emails


if __name__ == '__main__':
	main()
