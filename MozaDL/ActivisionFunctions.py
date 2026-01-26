import numpy as np

def tanh(X, Hadamard = False):
	
	if Hadamard:
		for x in range(len(X)):
			X[x] = np.tanh(X[x])
	else:
		X = np.tanh(X)
		
	return X


def ReLU(X, Hadamard = False):

	if Hadamard:
		for x in range(len(X)):
			if X[x] < 0: 
				X[x] = 0
	else:
		if X < 0:
			X = 0

	return X

def Sig(X, Hadamard = False):

	if Hadamard:
		for x in range(len(X)):
			X[x] = 1/ (1 + np.exp(-1*X[x]))

	else:
		X = 1/ (1 + np.exp(-1*X))

	return X