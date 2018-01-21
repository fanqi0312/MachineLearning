"""
手写识别
    逻辑回归
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
# Extracting MNIST_data/train-images-idx3-ubyte.gz
# Extracting MNIST_data/train-labels-idx1-ubyte.gz
# Extracting MNIST_data/t10k-images-idx3-ubyte.gz
# Extracting MNIST_data/t10k-labels-idx1-ubyte.gz

print("MNIST loaded")
print(trainimg.shape)
print(trainlabel.shape)
print(testimg.shape)
print(testlabel.shape)
# (55000, 784)
# (55000, 10)
# (10000, 784)
# (10000, 10)
# print (trainimg)
print(trainlabel[0]) # [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]

x = tf.placeholder("float", [None, 784]) # None is for infinite
y = tf.placeholder("float", [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax模型
# LOGISTIC REGRESSION MODEL
actv = tf.nn.softmax(tf.matmul(x, W) + b)
# 损失函数
# COST FUNCTION
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))
# OPTIMIZER
learning_rate = 0.01
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# PREDICTION
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
# ACCURACY
accr = tf.reduce_mean(tf.cast(pred, "float"))
# INITIALIZER
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()

arr = np.array([[31, 23, 4, 24, 27, 34],
                [18, 3, 25, 0, 6, 35],
                [28, 14, 33, 22, 20, 8],
                [13, 30, 21, 19, 7, 9],
                [16, 1, 26, 32, 2, 29],
                [17, 12, 5, 11, 10, 15]])

# tf.rank(arr).eval() 矩阵的维度，打印需要eval()
# tf.shape(arr).eval() 行列数
# tf.argmax(arr, 0).eval() 列上最大值的索引
# 0 -> 31 (arr[0, 0])
# 3 -> 30 (arr[3, 1])
# 2 -> 33 (arr[2, 2])
tf.argmax(arr, 1).eval() #行、上最大值的索引
# 5 -> 34 (arr[0, 5])
# 5 -> 35 (arr[1, 5])
# 2 -> 33 (arr[2, 2])

# 样本迭代50次
training_epochs = 50
batch_size = 100
display_step = 5
# SESSION
sess = tf.Session()
sess.run(init)
# MINI-BATCH LEARNING
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        # 梯度下降求解
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds) / total_batch
    # DISPLAY
    if epoch % display_step == 0:
        feeds_train = {x: batch_xs, y: batch_ys}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
              % (epoch, training_epochs, avg_cost, train_acc, test_acc))
print("DONE")



