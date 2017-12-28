# 每个图片8x8  识别数字：0,1,2,3,4,5,6,7,8,9

import numpy as np
from NeuralNetwork import NeuralNetwork
# 交叉运算 数据集 做拆分 训练集 测试集两部分
from sklearn.cross_validation import train_test_split
# 加载手写阿拉伯数字数据集
from sklearn.datasets import load_digits
# 对结果衡量的包
from sklearn.metrics import confusion_matrix, classification_report
# 用于转化二维数字类型
from sklearn.preprocessing import LabelBinarizer

# 下载数据集
digits = load_digits()
# 特征量
X = digits.data
y = digits.target
# 神经网络要求
X -= X.min()  # normalize the values to bring them into the range 0-1
X /= X.max()

# 输入层：跟特征向量相同（8X8），
# 隐藏层：有一定的灵活性 这个设计的比输入层多一些
# 输出层：类别结果：数字有0-9共10个，
nn = NeuralNetwork([64, 100, 10], 'logistic')

# 对数据类型转化为 0 1 形式 每种组合 是 sklearn的要求
# 交叉训练
X_train, X_test, y_train, y_test = train_test_split(X, y)
# 转化数据类型
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)

print("start fitting")
# 训练集特征向量 和 class label传入
nn.fit(X_train, labels_train, epochs=8000)
predictions = []
# 测试集每一行循环
for i in range(X_test.shape[0]):
    # 预测标签是多少
    o = nn.predict(X_test[i])
    # 结果0-1之间的值 选一个概率对应的整数
    predictions.append(np.argmax(o))

# y_test是预测 class label  predictions 是真实 class label
# 绘制准确率图标
print(confusion_matrix(y_test, predictions))

# 输出准确率
print(classification_report(y_test, predictions))
