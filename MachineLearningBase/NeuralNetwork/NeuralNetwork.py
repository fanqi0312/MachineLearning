"""
神经网络

"""
import numpy as np


# 神经网络类
class NeuralNetwork:
    """
    :param layers: 指定神经网络层数和神经元个数 最少有两层 input and output，如[2, 2, 1]
    :param activation: 激活函数可使用"logistic" 或 "tanh"（默认）
    """

    def __init__(self, layers, activation='tanh'):

        # 初始化-随机设置权重
        self.weights = []
        # 得到神经网络层数 排除第一层和最后一层循环
        for i in range(1, len(layers) - 1):
            # 对i层 与 i-1层 与第一层 进行权重(weight)连线赋值
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            # 对i层 与 i+1层 与最后一层 进行权重(weight)连线赋值
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

        # 初始化激活函数-使用"logistic" 或 "tanh"
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

    """ 
    训练神经网络
    :param X   训练集，二维矩阵
    :param y   分类标记
    :param learning_rate   学习率 数值大的话 步子大
    :param epochs  抽样计算10000次，每次样本随机抽取 全部执行运算较多 有一个数来控制次数
    """

    def fit(self, X, y, learning_rate=0.2, epochs=20000):
        ############# 处理X
        # 确认是二维数组
        X = np.atleast_2d(X)
        # ones初始化一个矩阵 参数是传入行数和列数+1 初始化的值全是1
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        # 取所有的行 ：列取第一列和除了最后一列
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        # #数据类型转换为 np科学计算数组格式
        y = np.array(y)

        ############# 训练神经网络（核心）
        for k in range(epochs):  # 循环次数
            # 随机抽取一行
            i = np.random.randint(X.shape[0])
            # 随机从x中抽取一个实例
            a = [X[i]]

            # 从输入层正向更新计算神经元中的值
            for l in range(len(self.weights)):  # going forward network, for each layer
                # Computer the node value for each layer (O_i) using activation function
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            # 误差值，根据真实的class lable 与预测 class lable 结果 做减法 output 计算error
            # Computer the error at the top layer
            error = y[i] - a[-1]

            # 反向更新，根据当前最后一层神经元的值进行反向更新
            # For output layer, Err calculation (delta is updated error)
            deltas = [error * self.activation_deriv(a[-1])]

            # 开始反向更新
            # Staring backprobagation
            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer 从最后一层开始到0层 每一次往回退一次
                # Compute the updated error (i,e, deltas) for each node going from top layer to input layer #更新隐藏层error更新 之前图解公式的Errj
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            # 顺序颠倒
            deltas.reverse()
            # 根据公式更新权重（weight）
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    # 预测结果 跟上面的正向流程类似计算出输出层的值 不需要保存每一层的值
    # 得数是0 到 1 有时候是 -1 到 1 以0.5为界限
    # 测试（与正向相似）
    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


# 直接调用一个简单的双曲函数
def tanh(x):
    return np.tanh(x)


# 求tanh导数function
def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


# 定义S型函数
def logistic(x):
    return 1 / (1 + np.exp(-x))


# 求logistic导数
def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))

"""
运行测试
"""
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
