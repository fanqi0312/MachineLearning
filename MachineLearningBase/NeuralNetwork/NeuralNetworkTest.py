from NeuralNetwork import NeuralNetwork
import numpy as np

# 实例化神经网络，2,2,1神经网络
nn = NeuralNetwork([2, 2, 1], 'tanh')

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

nn.fit(X, y)

for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print(i, nn.predict(i))
