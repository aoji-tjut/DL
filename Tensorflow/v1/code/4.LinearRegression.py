import tensorflow as tf
import numpy as np

#创建100个随机样本
x_data = np.random.random(size=100)
y_data = x_data*0.1 + 0.2
#构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b
#二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))#平均值(平方(误差))
#梯度下降法优化器 优化k、b接近于0.1、0.2
optimizer = tf.train.GradientDescentOptimizer(0.2)#学习率<1
#训练 最小化代价函数
train = optimizer.minimize(loss)#loss越小训练效果越好
#变量初始化
init = tf.global_variables_initializer()
#会话
with tf.Session() as sess:
    sess.run(init)
    for i in range(1001):
        sess.run(train)
        if i%100 == 0:
            print("次数：" + str(i) + "  ", "[k, b] = " + str(sess.run([k, b])))