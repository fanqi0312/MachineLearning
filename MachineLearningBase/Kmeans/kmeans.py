#-*-coding:utf-8-*-
'''
Created on 2017��4��27��

@author: mao
'''
import numpy as np

# kmeans算法（参数：maxIt最大循环次数）
def kmeans(X,k,maxIt):
    numPoints,numDim=X.shape
    dataSet=np.zeros((numPoints,numDim+1))

    #数据
    dataSet[:,:-1]=X
    # 分组结果（初始化分类）
    centroids=dataSet[np.random.randint(numPoints,size=k),:]
    centroids=dataSet[0:2, :] #强制中心点
    centroids[:,-1]=range(1,k+1)

    # 循环次数
    iterations=0
    # 旧中心点
    oldCentroids=None
    
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print("iterations:\n",iterations)
        print("dataSet:\n",dataSet)
        print("centroids:\n",centroids)
        # 不能用=，需要两份独立数据，而非指针变化
        oldCentroids=np.copy(centroids)
        iterations+=1

        # 重新分类
        updateLabels(dataSet, centroids)
        # 获取最新的中心点
        centroids=getCentroids(dataSet, k)
    
    return dataSet
    
    
# 停止条件
def shouldStop(oldCentroids,centroids,iterations,maxIt):
    # 超过最大循环次数
    if iterations>maxIt:
        return True
    # 中心点不在变化
    return np.array_equal(oldCentroids, centroids)

# 重新分类
def updateLabels(dataSet,centroids):
    numPoints,numDim=dataSet.shape
    for i in range(0,numPoints):
        dataSet[i,-1]=getLabelFromClosestCentroid(dataSet[i,:-1], centroids)
        
        
# 获取最近的中心点
def getLabelFromClosestCentroid(dataSetRow,centroids):
    label=centroids[0,-1]
    
    minDist=np.linalg.norm(dataSetRow-centroids[0,:-1])
    
    for i in range(1,centroids.shape[0]):
        dist=np.linalg.norm(dataSetRow-centroids[i,:-1])
        if dist<minDist:
            minDist=dist
            label=centroids[i,-1]
    print("minDist",minDist)
    return label

# 重新计算新的中心点
def getCentroids(dataSet,k):
    
    result=np.zeros((k,dataSet.shape[1]))
    for i in range(1,k+1):
        oneCluster=dataSet[dataSet[:,-1]==i,:-1]
        result[i-1,:-1]=np.mean(oneCluster,axis=0)
        result[i-1,-1]=i
    return result

x1=np.array([1,1])
x2=np.array([2,1])
x3=np.array([4,3])
x4=np.array([5,4])
testX=np.vstack((x1,x2,x3,x4))
result=kmeans(testX, 2, 300)
print("final result:\n",result)
    
    
    
    
    
    
    
    
    