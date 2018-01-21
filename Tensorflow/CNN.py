import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

print("number of train is %d" % (mnist.train.num_examples))
print("number of train is %d" % (mnist.test.num_examples))

n_input = 784
n_output = 10

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

stddev = 0.1
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=stddev)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=stddev)),
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 128, 1024], stddev=stddev)),
    'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=stddev))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64], stddev=stddev)),
    'bc2': tf.Variable(tf.random_normal([128], stddev=stddev)),
    'bd1': tf.Variable(tf.random_normal([1024], stddev=stddev)),
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=stddev))
}


def conv_basic(_input, _w, _b, _keepratio):
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])

    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)

    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)

    _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])

    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)

    _out = tf.add(tf.matmul(_fc1, _w['wd2']), _b['bd2'])

    out = {'input_r': _input_r,
           '_conv1': _conv1, '_pool1': _pool1, '_pool_dr1': _pool_dr1,
           '_conv2': _conv2, '_pool2': _pool2, '_pool_dr2': _pool_dr2,
           '_dense1': _dense1,
           '_fc1': _dense1, '_fc_dr1': _fc_dr1,
           'out': _out

           }
    return out


_pred = conv_basic(x, weights, biases, keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred, labels=y))
optm = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
init = tf.global_variables_initializer()

save_setp = 1
saver = tf.train.Saver(max_to_keep=2)

do_train = 0

sess = tf.Session()
sess.run(init)

training_epochs = 5
batch_size = 200
display_step = 1
if do_train == 1:
    for epochs in range(training_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)
        avg_cost = 0.
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            sess.run(optm, feed_dict={x: batch_x, y: batch_y, keepratio: 0.7})
            avg_cost += sess.run(cost, feed_dict={x: batch_x, y: batch_y, keepratio: 1.0}) / total_batch

        if epochs % display_step == 0:
            # epochs += 1
            train_acc = sess.run(accr, feed_dict={x: batch_x, y: batch_y, keepratio: 1.0})

            batch_x, batch_y = mnist.test.next_batch(batch_size)
            test_acc = sess.run(accr, feed_dict={x: batch_x, y: batch_y, keepratio: 1.0})

            print("Epochs: %03d/%03d cost:%.9f train acc:%.3f test acc:%.3f" % (epochs, training_epochs, avg_cost, train_acc, test_acc))
            saver.save(sess, "model/TensorflowCNN.ckpt-"+str(epochs))

if do_train == 0:
    epoch = training_epochs - 1
    saver.restore(sess, "model/TensorflowCNN.ckpt-"+str(epoch))

    total_batch = int(mnist.test.num_examples / batch_size)
    avg_acc = 0.
    for i in range(total_batch):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        test_acc = sess.run(accr, feed_dict={x: batch_x, y: batch_y, keepratio: 1.0})
        avg_acc += test_acc / total_batch

    print("Test acc : %.3f" % (avg_acc))
