#-*- coding:utf-8 -*-

import numpy as np
from scipy import signal


A=np.arange(9).reshape((3,3))

B=np.arange(4).reshape((2,2))
print("---------卷积结果---------------------")
X = signal.convolve(A,B, mode="valid")
print(X)
print("---------互相关结果---------------------")
Y = signal.correlate(A,B,mode='valid')
print(Y)
# --------------------- 
# 作者：俞驰的博客 
# 来源：CSDN 
# 原文：https://blog.csdn.net/appleyuchi/article/details/86574350 
# 版权声明：本文为博主原创文章，转载请附上博文链接！


from mxnet import nd
from mxnet.gluon import nn

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0]-p_h+1, X.shape[1]-p_w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i,j]= X[i:i+p_h, j:j+p_w].max()
            elif mode == 'avg':
                Y[i,j]= X[i:i+p_h, j:j+p_w].mean()
    return Y

X = nd.arange(9).reshape((3,3))
pool2d(X, (2,2))

pool2d(X, (2,2), mode='avg')