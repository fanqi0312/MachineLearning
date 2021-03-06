"""
使用训练好的模型
VGG有19层
1. 获得模型文件，并放到文件夹下
2. 加载模型文件
3. net函数


"""
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf


def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


print("Functions for VGG ready")


def net(data_path, input_image):
    # 定义参数-19的参数
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    # 加载模型
    data = scipy.io.loadmat(data_path)

    # 减均值（原模型减均值）
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    # 获取原模型的参数
    weights = data['layers'][0]
    # 实验模型的结构，逐层测试（如果不知道结构的话）
    # conv_1 w
    print(weights[0][0][0][0][0][0].shape)
    # conv_1 b
    print(weights[0][0][0][0][0][1].shape)

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        # 识别每层前4个字母，匹配
        kind = name[:4]

        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current
    assert len(net) == len(layers)
    return net, mean_pixel, layers

print("Network for VGG ready")

# 2. 加载模型文件

# 当前路径
cwd = os.getcwd()
VGG_PATH = cwd + "\\model\\imagenet-vgg-verydeep-19.mat"

IMG_PATH = cwd + "\\model\\dog.jpg"
input_image = imread(IMG_PATH)

shape = (1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
with tf.Session() as sess:
    image = tf.placeholder('float', shape=shape)
    nets, mean_pixel, all_layers = net(VGG_PATH, image)
    input_image_pre = np.array([preprocess(input_image, mean_pixel)])
    layers = all_layers  # For all layers
    # layers = ('relu2_1', 'relu3_1', 'relu4_1')
    for i, layer in enumerate(layers):
        print("[%d/%d] %s" % (i + 1, len(layers), layer))
        features = nets[layer].eval(feed_dict={image: input_image_pre})

        print(" Type of 'features' is ", type(features))
        print(" Shape of 'features' is %s" % (features.shape,))
        # Plot response
        if 1:
            plt.figure(i + 1, figsize=(10, 5))
            plt.matshow(features[0, :, :, 0], cmap=plt.cm.gray, fignum=i + 1)
            plt.title("" + layer)
            plt.colorbar()
            plt.show()
