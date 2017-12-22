# 每个图片8x8  识别数字：0,1,2,3,4,5,6,7,8,9

import numpy as np
# 下载数据集
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split

#下载数据集
digits = load_digits()
X = digits.data
y = digits.target
# 神经网络要求
X -= X.min()  # normalize the values to bring them into the range 0-1
X /= X.max()

# 输入层：跟特征向量相同（8X8），
# 隐藏层：灵活通常比输入层多
# 输出层：类别结果：数字有0-9共10个，
nn = NeuralNetwork([64, 100, 10], 'logistic')

# 交叉训练
X_train, X_test, y_train, y_test = train_test_split(X, y)
# 转化数据类型
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)

print("start fitting")
nn.fit(X_train, labels_train, epochs=8000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
