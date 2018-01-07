"""
手写识别
    简单神经网络

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# NETWORK TOPOLOGIES
# 网络结构：3层。（784-256-128-10）
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10

# INPUTS AND OUTPUTS
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
    out = (tf.matmul(layer_2, _weights['out']) + _biases['out'])
    return out


# PREDICTION
pred = multilayer_perceptron(x, weights, biases)

# LOSS AND OPTIMIZER
# 数交叉熵函数（损失函）softmax_cross_entropy_with_logits
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 梯度下降
optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
# 准确率
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# 转化为格式
accr = tf.reduce_mean(tf.cast(corr, "float"))

# INITIALIZER
init = tf.global_variables_initializer()
print("FUNCTIONS READY")

training_epochs = 20
batch_size = 100
display_step = 4
# LAUNCH THE GRAPH
sess = tf.Session()
sess.run(init)
# OPTIMIZE
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    # ITERATION
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost / total_batch
    # DISPLAY
    if (epoch + 1) % display_step == 0:
        print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys}
        train_acc = sess.run(accr, feed_dict=feeds)
        print("TRAIN ACCURACY: %.3f" % (train_acc))
        feeds = {x: mnist.test.images, y: mnist.test.labels}
        test_acc = sess.run(accr, feed_dict=feeds)
        print("TEST ACCURACY: %.3f" % (test_acc))
print("OPTIMIZATION FINISHED")
