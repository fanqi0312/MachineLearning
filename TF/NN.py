import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 如果数据没有会下载
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# print(mnist.train.num_examples)


n_input = 784
n_hidden_1 = 256
n_hidden_2 = 128
n_classes = 10

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

stddev = 0.1

weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def multilayer_perceptron(_x, _weights, _biases):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(_x, _weights['w1']), _biases['b1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, _weights['w2']), _biases['b2']))
    out = tf.add(tf.matmul(layer2, _weights['out']), _biases['out'])
    return out


pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, "float"))

init = tf.global_variables_initializer()
print("网络构造完成")

# 超函数

training_epochs = 100
batch_size = 100
display_step = 5

sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feeds = {x: batch_x, y: batch_y}

        sess.run(optm, feed_dict=feeds)

        avg_cost += sess.run(cost, feed_dict=feeds) / total_batch

    if (epoch) % display_step == 0:
        feeds = {x: batch_x, y: batch_y}
        train_acc = sess.run(accr, feed_dict=feeds)

        feeds = {x: mnist.test.images, y: mnist.test.labels}
        test_acc = sess.run(accr, feed_dict=feeds)

        print("Epoch: %03d/%03d cost:%.9f train_accuracy:%.3f test_accuracy:%.3f" % (epoch, training_epochs, avg_cost, train_acc, test_acc))
