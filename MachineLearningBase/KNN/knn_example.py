'''
Created on 20170424

@author: mao
'''
from sklearn import neighbors
from sklearn import datasets

# KNN分类器
knn=neighbors.KNeighborsClassifier()
# 加载内置训练集
iris=datasets.load_iris()

print(iris)

# fit建立模型
knn.fit(iris.data, iris.target)
# 创建测试集
predictedLabel=knn.predict([[0.1,0.2,0.3,0.4]])

print(predictedLabel)
# 0

