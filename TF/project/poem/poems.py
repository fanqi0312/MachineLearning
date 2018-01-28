import collections

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

"""
RNN结构

"""


def rnn_model(model,
              input_data,
              output_data,
              vocab_size,
              run_size=128,
              num_layers=2,
              batch_size=64,
              learning_rate=0.01
              ):
    end_points = {}

    # state_is_tuple新版本默认是True
    cell_fun = rnn.BasicLSTMCell


    cell = cell_fun(run_size, state_is_tuple=True)
    cell = rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, tf.float32)

    embedding = tf.Variable(tf.random_uniform([vocab_size + 1, run_size], -1.0, 1.0))
    inputs = tf.nn.embedding_lookup(embedding, input_data)

    # 推荐dynamic_rnn
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)

    weights = tf.Variable(tf.truncated_normal([run_size, vocab_size + 1]))
    bias = tf.Variable(tf.zeros(shape=vocab_size + 1))
    logits = tf.nn.add(tf.matmul(outputs, weights), bias=bias)

    if output_data is not None:
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = outputs
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state

    else:
        prediction = tf.nn.softmax(logits)
        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points


"""
数据预处理
"""


def process_poems(file_name):
    poems = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            title, content = line.strip().split(':')

            # 去除特殊符号的
            if '_' in content or '{' in content:
                continue
            # 去除长度大小的
            if len(content) < 5 or len(content) > 80:
                continue
            # 加前后标记
            content = start_token + content + end_token
            poems.append(content)

    # 排序
    poems = sorted(poems, key=lambda l: len(line))
    all_words = []
    for poem in poems:
        # 每个字（有重复）
        all_words += [word for word in poem]  # ??
    # 获取词频
    counter = collections.Counter(all_words)
    # 词出现的个数
    count_paris = sorted(counter.items(), key=lambda x: x[-1])

    words, _ = zip(*count_paris)
    # 取部分词，也可全部
    words = words[:len(words)]

    # 字转化为数字（用word2Vec效果更好）
    word_int_map = dict(zip(words, range(len(words))))

    # 映射为数字
    # poems_vector = [list(map(lambda word: word_int_map.get(word, len(words))))]
    poems_vector = list()
    for word in words:
        poems_vector.append(word_int_map.get(word))

    return poems_vector, word_int_map, words


"""
制作batch数据
"""


def generate_batch(batch_size, poems_vector, word_to_int):
    # 循环数量
    n_chunk = int(len(poems_vector) / batch_size)

    x_batchs = []
    y_batchs = []
    for i in range(n_chunk):
        # batch的索引范围
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(poems_vector))

        batches = poems_vector[start_index, end_index]

        # 空格填充数据,有5,7长度的
        length = max(map(len, batches))
        x_data = np.full((batch_size, length), word_to_int(' '), np.int32)

        for row in range(batch_size):
            x_data[row:len(batches[row])] = batches[row]
        # y是在x的基础上，平移一位
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]

        x_batchs.append(x_data)
        y_batchs.append(y_data)
    return x_batchs, y_batchs


tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size = ?')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning_rate = ?')
# 模型路径
tf.app.flags.DEFINE_string('check_point_dir', './model/', 'check_point_dir')
tf.app.flags.DEFINE_integer('check_point', 2, 'check_point')
# 文件路径
tf.app.flags.DEFINE_string('file_path', './data/poems.txt', 'file_path')

tf.app.flags.DEFINE_integer('epoch', 50, 'epoch')

# 训练的开始和结束标志
start_token = 'G'
end_token = 'E'

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def run_training():
    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
    batch_inputs, batch_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    # RNN模型

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(
        model='lstm',
        input_data=input_data,
        output_data=output_targets,
        vocab_size=len(vocabularies),
        run_size=128,
        num_layers=2,
        batch_size=64,
        learning_rate=0.01
    )

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        start_epoch = 0

        check_point = tf.train.latest_checkpoint(FLAGS.check_point)

        if check_point:
            saver.restore(sess, check_point)
            print("[INFO信息] restore fro the checkpoint {}".format(check_point))
            start_epoch += int(check_point.split('-')[-1])
        print('[INFO信息] start training...')

        # try:
        for epoch in range(start_epoch, FLAGS.epochs):
            n = 0
            n_chunk = len(poems_vector)
            for batch in range(n_chunk):
                loss, _, _ = sess.run([
                    end_points['total_loss'],
                    end_points['last_state'],
                    end_points['train_op']
                ], feed_dict={input_data: batch_inputs[n], output_targets: batch_outputs[n]})

                n += 1
                print('[INFO] %d, %d, %.6f ' % (epoch, batch, loss))

            if epoch % 6 == 0:
                saver.save(sess, './model/', global_step=epoch)


def main(is_train):
    if is_train:
        print('training')
        run_training()
    else:
        print('testing')
        begin_word = input('word')


# main
if __name__ == '__main__':
    main("training")
