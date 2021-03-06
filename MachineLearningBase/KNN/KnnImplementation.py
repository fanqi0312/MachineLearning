'''
kNN算法
算法：自己实现
数据集：文本格式，KnnImplementation.data.txt（注意命名规范）

'''
import csv
import random
import operator
import math 

# KNN原理代码

# 加载数据集（文件名，分割训练和测试）
def loadDataset(filename,split,trainset=[],testset=[]):
    # with open(filename,'rb')as csvfile:
    #     lines=csv.reader(csvfile)
    with open(filename, 'r') as csvfile:
        dataList = open(filename, 'r')
        lines = csv.reader(dataList)
        dataset=list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y]=float(dataset[x][y])
            if random.random()<split:
                trainset.append(dataset[x])
            else:
                testset.append(dataset[x])


# 计算两个实例的距离（支持多维度）参数：点1，点2，维度
def enclideanDistance(instance1,instance2,length):
    distance=0
    for x in range(length):
        # 分别计算每个维度  更好（A-B）2
        distance+=pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)


# 最近的几个邻居
def getNeighbors(trainset,testInstance,k):
    distance=[]
    length=len(testInstance)-1
    # 计算训练集每一个的距离
    for x in range(len(trainset)):
        dist=enclideanDistance(testInstance, trainset[x], length)
        distance.append((trainset[x],dist))
    # 排序
    distance.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors

# 邻居中相同类型最多的是什么
def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response=neighbors[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
    sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

# 准确率有多少
def getAccuracy(testSet,prediction):
    correct=0
    for x in range(len(testSet)):
        if testSet[x][-1]==prediction[x]:
            correct+=1
    return (correct/float(len(testSet)))*100
        

def main():
    trainSet=[]
    testSet=[]
    # 数据分割比例：2/3作为训练集，1/3为测试集
    split=0.7
    # 加载数据，并分割
    loadDataset(r'KnnImplementation.data.txt', split, trainSet, testSet)
    print('训练集Train set:'+repr(len(trainSet)))
    print('测试集Test set:'+repr(len(testSet)))


    prediction=[]
    # 临近数量
    k=3
    for x in range(len(testSet)):
        # 获取最近的3个点
        neighbors=getNeighbors(trainSet, testSet[x], k)
        # 根据临近点确定类型
        result=getResponse(neighbors)
        prediction.append(result)
    # 计算准确度
    accuracy=getAccuracy(testSet, prediction)
    print('准确率Accuracy:'+repr(accuracy) + '%')

main()