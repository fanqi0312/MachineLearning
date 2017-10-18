# -*- coding: utf-8 -*-
'''
Created on 20170425

@author: mao
'''
import numpy as np
import pylab as pl
from sklearn import svm

## 1.生成数据
# 随机seed，随机的结果不变。改变数值则换一套
np.random.seed(100)
# 正太分布产生。20点，2列，均值2，方差2.
X=np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
# 前20个是0，后20个是1
Y=[0]*20+[1]*20

# 2.建立模型
clf=svm.SVC(kernel='linear')
clf.fit(X,Y)

# 3.
w=clf.coef_[0]
# 斜率
a=-w[0]/w[1]
# 生成连续值
xx=np.linspace(-5,5)
#
yy=a*xx-(clf.intercept_[0])/w[1]


## 4.上下的切线
b=clf.support_vectors_[0]
yy_down=a*xx+(b[1]-a*b[0])
# 最后一个值
b=clf.support_vectors_[-1]
yy_up=a*xx+(b[1]-a*b[0])

print("w:",w)
print("a:",a)

print("support_vectors_",clf.support_vectors_)
print("clf.coef_",clf.coef_)

## 5.绘制图片
pl.plot(xx,yy,'k-')
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')
pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,-1],s=80,facecolors='none')
pl.scatter(X[:,0],X[:,-1],c=Y,cmap=pl.cm.Paired)    #scatter显示出离散的点
pl.axis('tight')
pl.show()








