"""
识别验证码

"""

import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from captcha.image import ImageCaptcha

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# def random_captcha_text(char_set=number+alphabet+ALPHABET, captcha_size=4):
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下  
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]  
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  
        return gray
    else:
        return img


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
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
        idx = i * CHAR_SET_LEN + int(c)
        vector[idx] = 1
    return vector


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


""" 
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有 
vec = text2vec("F5Sd") 
text = vec2text(vec) 
print(text)  # F5Sd 
vec = text2vec("SFd5") 
text = vec2text(vec) 
print(text)  # SFd5 
"""


# 生成一个训练batch
def get_next_batch(batch_size=128):
    # 0初始化
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

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
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


"""
定义CNN结构
参数：
w_alpha：0.01 使初始化权重值较小
b_alpha：0.1  
"""


def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3层 卷积conv layer

    # 3x3 1深度（灰度图是1） 特征图32
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    # 根特征图相同
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))

    # padding通常用SAME，pool不能整除，自动补零
    # relu激活，bias_add调用方法，conv2d构造卷积
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))

    """
    池化层
    max_pool（主流）
    ksize前后都为1，2x2窗口
    strides前后都为1，2x2，（与ksize配合）
    """
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # 第二层 经过一次池化层32翻倍64
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    """
    全连接层Fully connected layer
    参数：8 X 20，经过3层后的尺寸。分析尺寸，卷积没有改变大小，每次pool变为4/1，padding='SAME'不能整除，补全
    64 特征图数量，与最后的卷积相同
    1024向量
    
    """
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    # B与初始化向量相同。1024
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    # 激活，相加，矩阵相乘
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


"""
训练
"""


def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 保存模型
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            # 64个1个batch
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, loss_)

            # 每100 step计算一次准确率  
            if step % 10 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)

                # 如果准确率大于50%,保存模型,完成训练，global_step迭代名
                if acc > 0.50:
                    saver.save(sess, "./model/crack_capcha.model", global_step=step)
                    break

            step += 1


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/crack_capcha.model-810")

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        text = text_list[0].tolist()
        return text


"""
main函数


"""
if __name__ == '__main__':
    train = 1

    # 训练
    if train == 0:
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        # alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        # ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

        # 获取文字和图片
        text, image = gen_captcha_text_and_image()
        print("验证码图像channel:", image.shape)  # (60, 160, 3)  
        # 图像大小  
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        MAX_CAPTCHA = len(text)
        print("验证码文本最长字符数", MAX_CAPTCHA)
        # 文本转向量  
        # char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
        # 使用文字
        char_set = number
        CHAR_SET_LEN = len(char_set)

        # 固定输入规模60x160。彩色图转化为灰度图
        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        # 是40个。4x10。每个为0-9.
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
        #
        keep_prob = tf.placeholder(tf.float32)  # dropout

        # 定义网络结构
        train_crack_captcha_cnn()


    # 测试
    if train == 1:
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        char_set = number
        CHAR_SET_LEN = len(char_set)

        text, image = gen_captcha_text_and_image()

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(image)

        plt.show()

        MAX_CAPTCHA = len(text)
        image = convert2gray(image)
        image = image.flatten() / 255

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
        keep_prob = tf.placeholder(tf.float32)  # dropout

        predict_text = crack_captcha(image)
        print("正确: {}  预测: {}".format(text, predict_text))
