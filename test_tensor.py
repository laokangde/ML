# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 20:16:08 2019

@author: 123
"""

#%% 2次函数的简单线性回归

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-1, 1, 0.01)[:,np.newaxis]

#y = x**2 + (np.random.rand(x.shape[0], x.shape[1]) - 0.5 )*0.2
y = 2*x**2 + np.random.normal(0, 0.1, size=x.shape)

plt.scatter(x,y)
plt.show()

X = np.concatenate(( np.ones(shape=x.shape), x, x**2), axis=1) #生成数据矩阵X 第一列是ones

epsilon = 0.1
theta = np.random.rand(X.shape[1], 1) * 2*epsilon - epsilon #随机初始化系数
#theta = np.zeros((X.shape[1], 1))

m = X.shape[0]
J = 1/(2*m) * np.sum( (X @ theta - y)**2 ) #cost function
# 梯度下降
temp = np.zeros(shape=theta.shape)  # for store theta
alpha = 0.1                         #learning rate
iters = 500                         #
cost = np.zeros((1,iters))          #cost function

plt.ion() 

for iter in range(0, iters):
    for ii in range(0, theta.shape[0]):
        temp[ii, 0] = theta[ii, 0] - alpha * 1/m * np.sum( (X @ theta - y) * X[:, ii][:,np.newaxis] ) #batch gradient descent
    theta = temp.copy()
    cost[:, iter] = 1/(2*m) * np.sum( (X @ theta - y)**2 )
    if iter % 5 == 0:
        plt.cla()
        y_pre = X @ theta
        plt.scatter(x,y)
        plt.scatter(x, y_pre)
        plt.text(0.5, 0, 'Loss=%.4f' % cost[:, iter], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

#y_pre = X @ np.array([[0], [0], [1]])
#plt.scatter(np.arange(0,iters),cost)
plt.ioff()
plt.show()

#实现矩阵切片不丢失维度的几种方法
#X[:, ii][:,np.newaxis] X[ii, :][np.newaxis, :]
#X[:, ii, np.newaxis]  X[np.newaxis, ii, :]
#X[:, [ii] ]高维度可能存在问题







