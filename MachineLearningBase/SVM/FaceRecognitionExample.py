from __future__ import print_function

from  time import time
# 日志
import logging
# 人脸绘图工具
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

# print(__doc__)
# 日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
#########################################################

# 下载名人数据库（联网下载）.保存在：C:\Users\FAN\scikit_learn_data\lfw_home
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# 有多少个实例
n_samples, h, w = lfw_people.images.shape

# 提取特征向量值
x = lfw_people.data
# 矩阵第一列
n_features = x.shape[1]

y = lfw_people.target
# 实际名字
target_names = lfw_people.target_names
# 多少人做人脸识别
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples:%d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

#########################################################

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

#########################################################
# 组成元素
n_components = 150

print("Extracting the top %d eigenfaces from %d faces" % (n_components, x_train.shape[0]))
t0 = time()
# 数据降维
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(x_train)
print("done in %0.3fs" % (time() - t0))

# 从人脸照片提取特征值
eigenfaces = pca.components_.reshape((n_components, h, w))

print("ffff")
t0 = time()
# 转化降维数据
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
print("done in %0.3fs" % (time() - t0))

#########################################################

print("Fitting the classifier to the training set")
t0 = time()
# C惩罚权重，gamma核函数
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

# 建模
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(x_train_pca, y_train)

print("建模时间 %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

######################## 评估准确率 #################################
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(x_test_pca)
print("clf.predict in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


######################## 打印图形界面 #################################

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# 姓名标签
def title(y_pred, y_test, target_names, i):
    # 预测姓名
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    # 实际姓名
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted:%s\ntrue: %s' % (pred_name, true_name)


prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(x_test, prediction_titles, h, w)

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
# 特征向量脸
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
