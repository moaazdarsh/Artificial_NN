import nn
import numpy as np
import matplotlib.pyplot as plt



network = nn.NeuralNetwork([1, 9, 10, 1], ['tanh', 'tanh', None])

#X is a list of inputs
DataSpan = 30
X = [x for x in range(1,DataSpan)]
Y = [np.log(y) for y in range (1,DataSpan)]


network.Optimize(X, Y, LearningRate=0.001, epochs=400)

NetworkPredictions = [network.Run(x)[0] for x in X]

fig1, g1 = plt.subplots()
plt.title("NN Prediction vs Data")
g1.scatter([x for x in X], [y for y in Y], s=7 ,c="b", label="Data")
g1.plot([x for x in X], NetworkPredictions, "r", label="Network Prediction")
g1.legend()

g1.set_xlabel("Input")
g1.set_ylabel("Output")

'''
fig2, g2 = plt.subplots()
plt.title("Error Log")
network.ELog = network.ELog[2:]
g2.plot(range(len(network.ELog)), network.ELog, "g", label="Error Log")
'''
plt.show(block=False)



while True:

	I = input("Input:")
	if I == "":
		break

	I = I.split(",")
	for i in range(len(I)):
		I[i] = float(I[i])

	print(I)
	print(network.Run(I))