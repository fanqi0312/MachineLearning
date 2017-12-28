import numpy as np


class Network(object):

    # sizes: 每层神经元的个数
    #       例如: 第一层2个神经元,第二层3个神经元:：net = Network([2, 3, 1])

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # np.random.randn(y, 1): 随机从正态分布(均值0, 方差1)中生成
        # 偏差
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 权重 2*3=6个权重
        self.weights = [np.random.randn(y, x)
                        # zip：分别2、3层，和3、1层权重
                        for x, y in zip(sizes[:-1], sizes[1:])]


# net.weights[1] 存储连接第二层和第三层的权重 (Python索引从0开始数)


#     *
# *
#     *   *
# *
#     *

sizes = [2, 3, 1]
print("sizes", sizes[1:])  # [3, 1] 除去第一个

# 分别随机生成（3,1）和（1,1）
bias = [np.random.randn(y, 1) for y in sizes[1:]]  # [3, 1]
print("bias", bias)
# 等同于上面
bias1 = np.random.randn(3, 1)
bias2 = np.random.randn(1, 1)
bias = [bias1, bias2]
print("bias", bias)

# bias [array([[-0.6833749 ],
#      [-0.69356891],
#      [ 0.20814377]]), array([[-0.41492554]])]

# for x, y in zip(sizes[:-1], sizes[1:]):
#     print(x)

weights = [np.random.randn(y, x)  # 注意是y，x
           for x, y in zip(sizes[:-1], sizes[1:])]
print("zip", list(zip(sizes[:-1], sizes[1:])))
# zip [(2, 3), (3, 1)]

print("weights", weights)

weights1 = np.random.randn(3, 2)
weights2 = np.random.randn(1, 3)
weights = [weights1, weights2]
print("weights", weights)

# weights [array([[ 1.50884526, -0.17934346],
#        [-0.05034174,  1.6309417 ],
#        [ 1.63897873,  0.60461265]]), array([[-0.27657078, -0.2004241 , -0.23486909]])]

net = Network([2, 3, 1])
print("net.num_layers:", net.num_layers)  # 3
print("net.sizes:", net.sizes)  # [2, 3, 1]
print("net.biases:", net.biases)
# [array([[ 0.55361285],
#        [ 0.28001921],
#        [-2.30880359]]), array([[ 0.62804192]])]
print("net.weights:", net.weights)

# [array([[-1.64620603,  0.2461518 ],
#        [ 1.77034962,  0.60789845],
#        [-1.11021938,  0.93779691]]), array([[-0.8660257 , -0.7950434 ,  0.45783313]])]

"""
梯度下降算法

training_data：训练集
epochs：循环次数
mini_batch_size：每一小块的实例数
eta：学习率
test_data：测试集（默认空）
"""


def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    """
    Train the neural network using mini-batch stochastic
    gradient descent.  The "training_data" is a list of tuples
    "(x, y)" representing the training inputs and the desired
    outputs.  The other non-optional parameters are
    self-explanatory.  If "test_data" is provided then the
    network will be evaluated against the test data after each
    epoch, and partial progress printed out.  This is useful for
    tracking progress, but slows things down substantially.
    """
    # 获取测试集数量
    if test_data: n_test = len(test_data)
    # 获得训练集数量
    n = len(training_data)
    # 循环次数
    for j in range(epochs):
        #
        # shuffle随机数据顺序
        np.random.shuffle(training_data)
        mini_batches = [
            training_data[k:k + mini_batch_size]
            # 从0-n，每间隔mini_batch_size抽取数据
            for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            # 更新权重和偏向（重要）
            self.update_mini_batch(mini_batch, eta)
        # 每轮评估一次训练效果
        if test_data:
            # 轮数、准确度、测试数量
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
        else:
            print("Epoch {0} complete".format(j))


"""
mini_batch:每一小块的实例数
eta:学习率
"""


def update_mini_batch(self, mini_batch, eta):
    """
    Update the network's weights and biases by applying
    gradient descent using backpropagation to a single mini batch.
    The "mini_batch" is a list of tuples "(x, y)", and "eta"
    is the learning rate.
    """
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    # 更新权重
    self.weights = [w - (eta / len(mini_batch)) * nw
                    for w, nw in zip(self.weights, nabla_w)]
    # 更新偏向
    self.biases = [b - (eta / len(mini_batch)) * nb
                   for b, nb in zip(self.biases, nabla_b)]
