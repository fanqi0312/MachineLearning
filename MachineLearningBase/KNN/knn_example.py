'''
kNN算法实例

使用neighbors
数据集：datasets

'''
from sklearn import neighbors
from sklearn import datasets

# KNN分类器
knn=neighbors.KNeighborsClassifier()
# 加载内置训练集
iris=datasets.load_iris()

print(iris)
# {'data': array([[ 5.1,  3.5,  1.4,  0.2],
#        [ 4.9,  3. ,  1.4,  0.2],
#        [ 5.4,  3.9,  1.7,  0.4],
# 。。。。
# 'target': array([0, 0, 0, 1, 1, 2, 2 2, 2]), 'target_names': array(['setosa', 'versicolor', 'virginica'],

# fit建立模型
knn.fit(iris.data, iris.target)
# 创建测试集
predictedLabel=knn.predict([[0.1,0.2,0.3,0.4]])

print(predictedLabel)
# 0

