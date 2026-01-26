import numpy as np
import matplotlib.pyplot as plt

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
class NeuralNetwork:
	#"Layers" is the dimensions of the network
	#NeuralNetwork.nn contains the weights and biases
	def __init__(self, Layers, ActivationFunction):
		self.Layers = Layers
		self.nn = []
		self.ActivationFunction = ActivationFunction

		self.DataMean = None
		self.DataStd = None

		self.ELog = []

		for l in range(len(self.Layers)-1):

			NextLayer = self.Layers[l+1]

			weights = np.random.uniform(-1, 1, size=(NextLayer, self.Layers[l]))
			biases = np.random.uniform(-1, 1, size=NextLayer)

			self.nn.append([weights, biases])



	def Run(self, Inputs, ReturnAnZ = False):
		Inputs = np.array([Inputs])

		Inputs = (Inputs - self.DataMean)/self.DataStd #Normalization

		A = [Inputs]
		Z = []
    
		LayerActivation = Inputs

		for l in range(len(self.nn)):
			
			LayerActivation = np.matmul(self.nn[l][0], LayerActivation) + self.nn[l][1]

			Z.append(LayerActivation)

			if (self.ActivationFunction[l] != None) & (l != 0):	
				match self.ActivationFunction[l].lower():
					case "relu":
						LayerActivation = ReLU(LayerActivation, True)
					case "sig":
						LayerActivation = Sig(LayerActivation, True)
					case "tanh":
						LayerActivation = tanh(LayerActivation, True)


			A.append(LayerActivation)
		A.pop(-1)

		LayerActivation = LayerActivation * self.DataStd + self.DataMean #Denormalization


		#A contains the activations of all nodes except output
		#Z contains all the weighted sums before the activation function
		if ReturnAnZ:
			return(LayerActivation, A, Z)
		else:
			return(LayerActivation)
		


	def Optimize(self, X, Y, LearningRate = 0.0005, epochs = 300):

		self.DataMean = np.mean(X, axis=0)
		self.DataStd = np.std(X, axis=0)
		
		for epoch in range(epochs):
			
			if epoch % 10 == 0 and epoch != 0:
				print(f"{epoch}/{epochs}")
			
			EpochPs = []
			for x,y in zip(X,Y):

				P, A, Z = self.Run(x, True)

				EpochPs.append(P) #the prediction over the data in the current epoch

				El = None
				#backpropagation algorithm
				LayerIndex = len(self.nn) - 1
				OutLayer = True
				while 0 <= LayerIndex:

					if OutLayer:
						#El= Ea ⊙ f′(zL), where l is the last layer and Ea is a vector of the dEda of l
						El = -2 * (y - P)

					else:
						#the equation for the rate of change of Error relative to the activations of a layer.
						#El = ((Wl+1)T x El+1) ⊙ f′(zl)		
						El = np.matmul(self.nn[LayerIndex + 1][0].transpose(), El)

					if (self.ActivationFunction[LayerIndex] != None) & (LayerIndex != 0):
						dfdZl = []	
						match self.ActivationFunction[LayerIndex].lower():
						#Multiply El by the derivative of f in respect to the weighted sums of the layer l.
							case "relu":
								for z in Z[LayerIndex]:
									if z < 0:
										dfdZl.append(0)
									else:
										dfdZl.append(1)

							case "sig":
								if not OutLayer:
									for a in A[LayerIndex+1]:
										dfdZl.append(a * (1-a))
								else:
									for a in P:
										dfdZl.append(a * (1-a))

							case "tanh":
								if not OutLayer:
									for a in A[LayerIndex+1]:
										dfdZl.append(1 - a**2)
								else:
									for a in P:
										dfdZl.append(1 - a**2)
							
						El = El * dfdZl

					#optimizing the biases
					El = np.clip(El, -5, 5)
					self.nn[LayerIndex][1] -= El * LearningRate

					#Calculating the error in weights and optimizing them
					# dEdW = El (column vector) × A[LayerIndex] (row vector)
					dEdW = np.outer(El, A[LayerIndex])
					dEdW = np.clip(dEdW, -3, 3)
					self.nn[LayerIndex][0] -= dEdW * LearningRate
					LayerIndex -= 1
					OutLayer = False
			
			self.ELog.append(0.5 * np.mean((np.array(Y) - np.array(EpochPs))**2))
