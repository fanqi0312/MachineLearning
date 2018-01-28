import re

import numpy as np


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


"""
加载数据
    参数：好地址，坏地址
    返回：文本（合并），标签（好01，坏10）
"""
def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    positive = open(positive_data_file, "rb").read().decode('utf-8')
    negative = open(negative_data_file, "rb").read().decode('utf-8')

    # 按照回车分割，排除最后一个[:-1]
    positive_examples = positive.split('\n')[:-1]
    negative_examples = negative.split('\n')[:-1]

    # 去空格
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]

    # positive_examples = list(open(positive_data_file, "rb").read().decode('utf-8'))
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open(negative_data_file, "rb").read().decode('utf-8'))
    # negative_examples = [s.strip() for s in negative_examples]
    # Split by words

    # 组合数据
    x_text = positive_examples + negative_examples
    # 清洗数据，过滤无用符号
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    # 生成标签，分两类
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    # 链接label
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

"""
获取迭代batch
"""
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            # 超过最大值，则取最大值
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
