'''
Created on 20170426

@author: mao
'''
from numpy import genfromtxt
import numpy as np
from sklearn import datasets,linear_model

############读取文件数据
# 读取数据，r忽略特殊字符格式
dataPath=r"MultipleRegression_data.csv"
# 转化为numpyArray矩阵格式，分割符号是，
deliverData=genfromtxt(dataPath,delimiter=',')
print("data")
print(deliverData)

############提取X，Y
# 所有行，开始到倒数第一列（不包含最后一列）：除最后一列
X=deliverData[:,:-1]
# 所有行，倒数第一列。只取最后一列
Y=deliverData[:,-1]
print("X:")
print(X)
print("Y:")
print(Y)

############执行函数
# 调用线性回归模型
regr=linear_model.LinearRegression()
regr.fit(X, Y)
print("coefficients:")
print(regr.coef_)

print("intercept:")
print(regr.intercept_)

############预测
xPred=[102,6]
yPred=regr.predict(xPred)
print("yPred:")
print(yPred)





