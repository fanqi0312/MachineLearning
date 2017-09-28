# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:49:09 2017

@author: Fanqi
"""

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO


# 1.读取训练集，获取特征值和标签值
with open(r'training.csv', 'r') as allElectronicsData:
    allElectronicsData = open(r'training.csv', 'r')
    reader = csv.reader(allElectronicsData)
    headers = next(reader)
    print("featureList:" + str(headers))
    # ['RID', 'age', 'income', 'student', 'credit_rating', 'class_buy_computer']
#收集特征值
featureList = []
#收集标签值
labelList = []
for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
print("featureList:" + str(featureList))
# [{'age': 'youth ', 'income': 'high ', 'student': 'no ', 'credit_rating': 'fair '}, {'age': 'youth ', 'income': 'high ', 'student': 'no ', 'credit_rating': 'excellent '}, {'age': 'middle_aged', 'income': 'high ', 'student': 'no ', 'credit_rating': 'fair '}, {'age': 'senior ', 'income': 'medium ', 'student': 'no ', 'credit_rating': 'fair '}, {'age': 'senior ', 'income': 'low ', 'student': 'yes ', 'credit_rating': 'fair '}, {'age': 'senior ', 'income': 'low ', 'student': 'yes ', 'credit_rating': 'excellent '}, {'age': 'middle_aged', 'income': 'low ', 'student': 'yes ', 'credit_rating': 'excellent '}, {'age': 'youth ', 'income': 'medium ', 'student': 'no ', 'credit_rating': 'fair '}, {'age': 'youth ', 'income': 'low ', 'student': 'yes ', 'credit_rating': 'fair '}, {'age': 'senior ', 'income': 'medium ', 'student': 'yes ', 'credit_rating': 'fair '}, {'age': 'youth ', 'income': 'medium ', 'student': 'yes ', 'credit_rating': 'excellent '}, {'age': 'middle_aged', 'income': 'medium ', 'student': 'no ', 'credit_rating': 'excellent '}, {'age': 'middle_aged', 'income': 'high ', 'student': 'yes ', 'credit_rating': 'fair '}, {'age': 'senior ', 'income': 'medium ', 'student': 'no ', 'credit_rating': 'excellent '}]
print("labelList:" + str(labelList))
# ['no ', 'no ', 'yes ', 'yes ', 'yes ', 'no ', 'yes ', 'no ', 'yes ', 'yes ', 'yes ', 'yes ', 'yes ', 'no ']
allElectronicsData.close()


# 2.转化数据为sklearn要求的数据
# 转化featureList
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print("dummyX:" + str(dummyX))
print("vec.get_feature_names():" + str(vec.get_feature_names()))

# 转化labelList
print("labelList:" + str(labelList))
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY:" + str(dummyY))


## 3.决策树处理
# 指定决策树算法
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf:" + str(clf))


## 4.将决策树输出为dot文件
with open("DecisionTree.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)


## 5.给定新的数据进行预测
# 创建测试数据
oneRowX = dummyX[0, :]
print("oneRowX:" + str(oneRowX))
newRowX = oneRowX
# 修改数据
newRowX[0] = 1
newRowX[2] = 0
print("newRowX:" + str(newRowX))
# 使用决策树测试
predictedY = clf.predict(newRowX)
print("predictedY:" + str(predictedY))
#购买1