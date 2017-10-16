'''
Created on 20170425

@author: mao
'''
from sklearn import svm
x=[[2,0],[1,1],[2,3],[3,4],[3,5]]
y=[0,0,1,1,1]

clf=svm.SVC(kernel='linear')
# 建立模型
clf.fit(x,y)

print(clf)
# 支持向量点
print(clf.support_vectors_)
# 支持向量点在数组中的位置
print(clf.support_)
#
print(clf.n_support_)

print(clf.predict([2, .0]))