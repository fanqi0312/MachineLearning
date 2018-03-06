"""
NN神经网络
    Tensorflow实现

读取数据集
设计网络结构784-256-128-10

输入和输出变量
初始化权重weights和偏置biases（数组）,stddev

核心计算
    前向传播计算multilayer_perceptron,矩阵乘积+B，激活
    平均损失函数：预测与实际对比
    梯度下降(学习率)
准确的平均值（boolean）
初始化为数值0,1，平均值

定义超函数
循环
    分割数据集

    循环-训练
        获取每次数据
        优化求解
        计算损失

    打印-测试
        训练集测试
        测试集测试




"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:\Workspace\Python\MachineLearning\Data\MNIST", one_hot=True)

# NETWORK TOPOLOGIES
# 网络结构：3层。（784-256-128-10）
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 128
n_classes = 10

# INPUTS AND OUTPUTS
# 运行时赋值， None最大化（不知道多少）
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# NETWORK PARAMETERS
# 高斯初始化参数
stddev = 0.1
# 权重数：相邻两层神经元相乘
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
}
# 偏置数：相邻两层神经元相乘
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
print("NETWORK READY")


# 前向传播
def multilayer_perceptron(_X, _weights, _biases):
    # tf.nn.sigmoid激活函数

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['w2']), _biases['b2']))
    # 输出层不需要sigmoid
    out = tf.matmul(layer_2, _weights['out']) + _biases['out']
    return out


# PREDICTION
pred = multilayer_perceptron(x, weights, biases)

# LOSS AND OPTIMIZER
# 损失函（交叉熵函数）softmax_cross_entropy_with_logits
# 参数：
# logits：预测结果
# labels：实际结果
# tf.reduce_mean平均值。loss。
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 梯度下降
optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

###### 准确率
# 预测对比真实，索引是否相同。结果是true，false
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# 将boolean转化为float 0,1
# reduce_mean 求均值
accr = tf.reduce_mean(tf.cast(corr, "float"))

# INITIALIZER
init = tf.global_variables_initializer()
print("FUNCTIONS READY")

# 超参数
training_epochs = 50
batch_size = 100
display_step = 4

# LAUNCH THE GRAPH


sess = tf.Session()
sess.run(init)
# OPTIMIZE
for epoch in range(training_epochs):
    avg_cost = 0.
    # 分割数据，batch数量
    total_batch = int(mnist.train.num_examples / batch_size)
    # ITERATION
    for i in range(total_batch):
        # 获取每次迭代数据
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        # 优化求解
        sess.run(optm, feed_dict=feeds)
        # 计算平均损失值
        avg_cost += sess.run(cost, feed_dict=feeds) / total_batch


    # DISPLAY 测试一次
    if (epoch + 1) % display_step == 0:
        # print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys}
        train_acc = sess.run(accr, feed_dict=feeds)
        # print("TRAIN ACCURACY: %.3f" % (train_acc))
        feeds = {x: mnist.test.images, y: mnist.test.labels}
        test_acc = sess.run(accr, feed_dict=feeds)
        # print("TEST ACCURACY: %.3f" % (test_acc))
        print("Epoch: %03d/%03d cost: %.9f train_accuracy: %.3f test_accuracy: %.3f" % (epoch, training_epochs, avg_cost, train_acc, test_acc))
print("OPTIMIZATION FINISHED")
