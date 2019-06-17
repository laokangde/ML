# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 20:16:08 2019

@author: 123
"""

import tensorflow as tf

#%% Session()
m1 = tf.constant([[3, 3]])
m2 = tf.constant([[2], 
                 [2]])
product = tf.matmul(m1, m2)
# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2 
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
    result = sess.run(m1)
    print(result)

#%% Variable()
var = tf.Variable(0,name='counter')    # our first variable in the "global_variable" set
print(var.name)
one = tf.constant(1)

add_operation = tf.add(var, one)
update_operation = tf.assign(var, add_operation)

#init = tf.initialize_all_variables() #被弃用

with tf.Session() as sess:
#    sess.run(init)
    # once define variables, you have to initialize them by doing this
    sess.run(tf.global_variables_initializer())# 初始化变量
    for _ in range(3):
        sess.run(update_operation)
        print(sess.run(var))   

#%% placeholder 外界传入数据
import tensorflow as tf

x1 = tf.placeholder(dtype=tf.float32, shape=None)
y1 = tf.placeholder(dtype=tf.float32, shape=None)
z1 = x1 + y1

x2 = tf.placeholder(dtype=tf.float32, shape=[2, 1])
y2 = tf.placeholder(dtype=tf.float32, shape=[1, 2])
z2 = tf.matmul(x2, y2)

with tf.Session() as sess:
    # when only one operation to run
    z1_value = sess.run(z1, feed_dict={x1: 1, y1: 2})

    # when run multiple operations
    z1_value, z2_value = sess.run(
        [z1, z2],       # run them together
        feed_dict={
            x1: 1, y1: 2,
            x2: [[2], [2]], y2: [[3, 3]]
        })
    print(z1_value)
    print(z2_value)

#%% 简单回归
    
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise                          # shape (100, 1) + some noise

# plot data
plt.scatter(x, y)
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)     # input x
tf_y = tf.placeholder(tf.float32, y.shape)     # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer 10个神经元
output = tf.layers.dense(l1, 1)                     # output layer

loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

plt.ion()   # something about plotting 继续plot

for step in range(100):
    # train and net output
    _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
    if step % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()


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







