"""
手写识别
    CNN神经网络

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print("MNIST ready")

n_input = 784
n_output = 10
weights = {
    # 卷基层参数说明（4个）：H W 深度 特征值
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
    # 卷基层）：64是上一层的特征值
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1)),
    #
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 128, 1024], stddev=0.1)),
    'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=0.1))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
}


def conv_basic(_input, _w, _b, _keepratio):
    # INPUT
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
    # CONV LAYER 1
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    # _mean, _var = tf.nn.moments(_conv1, [0, 1, 2])
    # _conv1 = tf.nn.batch_normalization(_conv1, _mean, _var, 0, 1, 0.0001)
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    # CONV LAYER 2
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    # _mean, _var = tf.nn.moments(_conv2, [0, 1, 2])
    # _conv2 = tf.nn.batch_normalization(_conv2, _mean, _var, 0, 1, 0.0001)
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
    # VECTORIZE
    _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])
    # FULLY CONNECTED LAYER 1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    # FULLY CONNECTED LAYER 2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
    # RETURN
    out = {'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
           'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
           'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
           }
    return out


print("CNN READY")

a = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1))
print(a)
a = tf.Print(a, [a], "a: ")
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# sess.run(a)


# print (help(tf.nn.conv2d))
print(help(tf.nn.max_pool))

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# FUNCTIONS

_pred = conv_basic(x, weights, biases, keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred, labels=y))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
init = tf.global_variables_initializer()

# 保存模型
# 每个都保存
save_step = 1
# max_to_keep：只保留最后2组模型
saver = tf.train.Saver(max_to_keep=2)

print("GRAPH READY")

# 是否训练测试，1为训练，0为测试
do_train = 0


sess = tf.Session()
sess.run(init)

training_epochs = 15
batch_size = 16
display_step = 1
if do_train == 1 :
    for epoch in range(training_epochs):
        avg_cost = 0.
        # total_batch = int(mnist.train.num_examples/batch_size)
        total_batch = 10
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio: 0.7})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.}) / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.})
            print(" Training accuracy: %.3f" % (train_acc))
            # test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio:1.})
            # print (" Test accuracy: %.3f" % (test_acc))

            # 保存网络Save Net
            if epoch % save_step == 0:
                saver.save(sess, "model/TensorFlowCNN.ckpt-" + str(epoch))

        print("OPTIMIZATION FINISHED")

if do_train == 0:
    # 读取最后一个模型
    epoch = training_epochs - 1
    saver.restore(sess, "model/TensorFlowCNN.ckpt-" + str(epoch))

    test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio: 1.})
    print(" TEST ACCURACY: %.3f" % (test_acc))