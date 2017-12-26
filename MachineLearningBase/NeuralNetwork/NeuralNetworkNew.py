"""
神经网络(支持3层)6.2

"""
import numpy as np


# 神经网络类
class NeuralNetworkNew:
    """
    :param layers: 数列，包含每层神经元的数量，如[2, 2, 1]
    :param 激活函数: 可使用"logistic" or "tanh"（默认）
    """

    def __init__(self, layers, activation='tanh'):

        # 初始化-随机设置权重
        self.weights = []
        # 从第二层到最后一层
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

        # 初始化激活函数-使用双曲函数或逻辑函数
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

    """ 
    训练神经网络
    :param X   训练集
    :param y   分类标记
    :param learning_rate   学习率
    :param epochs  抽样计算10000次
    """

    def fit(self, X, y, learning_rate=0.2, epochs=20000):
        ############# 处理X，x增加一列，值为1
        # 确认是二维
        X = np.atleast_2d(X)
        # 生成1.0的2维矩阵（行数，列数+1）
        X0 = X.shape[0]
        X1 = X.shape[1] + 1
        temp = np.ones([X0, X1])
        # 第一列和除最后一列
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp

        ############# 处理Y
        # 转化标准array
        y = np.array(y)

        ############# 训练神经网络（核心）
        for k in range(epochs):  # 训练次数
            # 从实例中随机取一个
            X0 = X.shape[0]
            i = np.random.randint(X0)
            Xi = X[i]
            a = [Xi] #实例

            #
            lenWeights = len(self.weights)
            for l in range(lenWeights):  # going forward network, for each layer
                # Computer the node value for each layer (O_i) using activation function
                wl = self.weights[l]
                al = a[l]
                npDot = np.dot(al, wl)  # 加权求和
                activationDot = self.activation(npDot)
                a.append(activationDot)

            # 误差值
            error = y[i] - a[-1]  # Computer the error at the top layer

            # 反向更新
            # For output layer, Err calculation (delta is updated error)
            a_1 = a[-1]
            activationa1 = self.activation_deriv(a_1)
            deltas = [error * activationa1]

            # Staring backprobagation
            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer
                # Compute the updated error (i,e, deltas) for each node going from top layer to input layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    # 测试（与正向相似）
    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


# tanh函数
def tanh(x):
    return np.tanh(x)


# tan函数倒数
def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


# 逻辑函数
def logistic(x):
    return 1 / (1 + np.exp(-x))


# 逻辑函数导数
def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


nn = NeuralNetworkNew([2, 2, 1], 'tanh')

# 训练集
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

# 训练
nn.fit(X, Y, 0.3, 30000)

# 测试集
# for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
for i in X:
    print(i, nn.predict(i))
