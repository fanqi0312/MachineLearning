'''
Created on 20170426

@author: mao
'''
import numpy as np
import random

# 参数：x矩阵，alpha学习率，m多少个实例，numIterations更新多少次

def gradientDescent(x,y,theta,alpha,m,numIterations):
    # 转置矩阵
    xTrans=x.transpose()
    # 循环次数
    for i in range(0,numIterations):
        hypothesis=np.dot(x,theta)
        # 偏差，估计-实际
        loss=hypothesis-y
        #
        cost=np.sum(loss**2)/(2*m)
        print("Iteration %d / Cost %f"%(i,cost))
        #
        gradient=np.dot(xTrans,loss)/m
        #
        theta=theta-alpha*gradient
    return theta

# 参数：多少行，偏差，方差
def genData(numPoints,bias,variance):
    # numPoints行，2列，值为0的矩阵
    x=np.zeros(shape=(numPoints,2))
    # numPoints行，1列，值为0的矩阵
    y=np.zeros(shape=numPoints)

    # range包含0，不包含最后
    for i in range(0,numPoints):
        x[i][0]=1
        x[i][1]=i
        # +偏差+随机值
        y[i]=(i+bias)+random.uniform(0,1)*variance
    return x,y


x,y=genData(100, 25, 10)

print("x:")
print(x)
print("y:")
print(y)



m,n=np.shape(x)
m_y=np.shape(y)
print("x shape:",str(m),",",str(n))
print("y shape:",str(m_y))

numIterations=20000
alpha=0.0005
theta=np.ones(n)
theta=gradientDescent(x, y, theta, alpha, m, numIterations)
print("theta",theta)
