import random

import numpy as np
import tensorflow as tf
from PIL import Image
from captcha.image import ImageCaptcha

captcha_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# def gen_captcha_text_and_image():

def random_captcha_text(captcha_set1=captcha_set, size=4):
    captcha_text = []

    for i in range(size):
        c = random.choice(captcha_set1)
        captcha_text.append(c)

    return captcha_text


"""
根据文字生成图片
"""


def gen_captcha_text_and_image():
    # 验证码生成类
    image = ImageCaptcha()

    # 调用-获取验证码文字
    captcha_text = random_captcha_text()
    # List转化为字符串
    captcha_text = ''.join(captcha_text)

    # 生成图片
    captcha = image.generate(captcha_text)

    # 保存到磁盘
    # image.write(captcha_text, captcha_text + '.jpg')

    captcha_image = Image.open(captcha)
    # 转化为np.array（供Tensorflow识别）
    captcha_image = np.array(captcha_image)
    # 返回label，和图片
    return captcha_text, captcha_image


def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, image_height, image_width, 1])

    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding="SAME"), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding="SAME"), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding="SAME"), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv3 = tf.nn.dropout(conv3, keep_prob)

    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    w_b = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), w_b))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, max_captcha * captcha_size]))
    b_out = tf.Variable(b_alpha * tf.random_normal([max_captcha * captcha_size]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


# 向量转回文本
def vec2text(vec):
    """
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx <36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    """
    text = []
    char_pos = vec.nonzero()[0]
    for i, c in enumerate(char_pos):
        number = i % 10
        text.append(str(number))

    return "".join(text)


def text2vec(text):
    text_len = len(text)
    if text_len > max_captcha:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(max_captcha * captcha_size)
    """
    def char2pos(c):  
        if c =='_':  
            k = 62  
            return k  
        k = ord(c)-48  
        if k > 9:  
            k = ord(c) - 55  
            if k > 35:  
                k = ord(c) - 61  
                if k > 61:  
                    raise ValueError('No Map')   
        return k  
    """
    for i, c in enumerate(text):
        idx = i * captcha_size + int(c)
        vector[idx] = 1
    return vector


def get_next_batch(batch_size=128):
    # 0初始化
    batch_x = np.zeros([batch_size, image_height * image_width])
    batch_y = np.zeros([batch_size, max_captcha * captcha_size])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    # 循环64个样本
    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        # 图像转化为灰度图
        image = convert2gray(image)

        # 图像255 归一化
        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        # 数值转化为label向量
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def train_crack_captcha_cnn():
    output = crack_captcha_cnn()

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, max_captcha, captcha_size])
    max_idx_p = tf.argmax(predict, 2)
    la_predict = tf.reshape(Y, [-1, max_captcha, captcha_size])
    max_idx_l = tf.argmax(la_predict, 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(100)
            # 同时计算
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 20 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)

                # 如果准确率大于50%,保存模型,完成训练，global_step迭代名。
                # 有问题，如果出现个别超过0.5就停止了
                if acc > 0.80 or step == 10000:
                    saver.save(sess, "./model/crack_capcha.model", global_step=step)
                    break

            step += 1


if __name__ == '__main__':
    captcha_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    captcha_size = 4
    captcha_text = random_captcha_text(captcha_set, captcha_size)
    print(captcha_text)

    image_height = 60
    image_width = 160
    max_captcha = len(captcha_set)

    X = tf.placeholder(tf.float32, [None, image_height * image_width * 1])
    Y = tf.placeholder(tf.float32, [None, captcha_size * max_captcha])
    keep_prob = tf.placeholder(tf.float32)

    train_crack_captcha_cnn()
