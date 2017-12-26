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
bias = [np.random.randn(y, 1) for y in sizes[1:]]
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

net = Network([2,3,1])
print("net.num_layers:", net.num_layers) # 3
print("net.sizes:", net.sizes) # [2, 3, 1]
print("net.biases:", net.biases)
# [array([[ 0.55361285],
#        [ 0.28001921],
#        [-2.30880359]]), array([[ 0.62804192]])]
print("net.weights:", net.weights)
# [array([[-1.64620603,  0.2461518 ],
#        [ 1.77034962,  0.60789845],
#        [-1.11021938,  0.93779691]]), array([[-0.8660257 , -0.7950434 ,  0.45783313]])]