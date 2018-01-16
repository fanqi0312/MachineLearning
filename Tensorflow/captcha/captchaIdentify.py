"""
生成验证码

"""

import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha

# 生成验证码范围
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

"""
生成验证码文字
# char_set：生成验证码范围
# captcha_size：数量，4的范围就可以。太大了训练量比较大
"""


# def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=5):
def random_captcha_text(char_set=number, captcha_size=5):
    captcha_text = []
    # 循环4次
    for i in range(captcha_size):
        # 随机选择1个
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


"""
根据文字生成图片
"""


def gen_captcha_text_and_image():
    # 验证码生成类
    image = ImageCaptcha()

    # 获取验证码文字
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


"""
main函数
绘制并显示验证码
"""
if __name__ == '__main__':
    # 获取文字和图片
    text, image = gen_captcha_text_and_image()

    # 绘制文字
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)

    # 绘制图片
    plt.imshow(image)

    # 显示
    plt.show()
