"""
测试神经网络

"""
from NeuralNetwork import NeuralNetwork
import numpy as np

# 实例化神经网络，2,2,1神经网络
nn = NeuralNetwork([2, 2, 1], 'tanh')

# 训练集
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

# 训练
nn.fit(X, Y, 0.3, 30000)

# 测试集
# for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
for i in X:
    print(i, nn.predict(i))
