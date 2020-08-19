import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#样本
x_data = np.linspace(-0.5, 0.5, 200).reshape(200, -1)
noise = np.random.normal(-0.01, 0.01, x_data.shape)#形状同x_data
y_data = np.square(x_data) + noise
#根据样本定义两个占位符
x = tf.placeholder(tf.float32, [None, 1])#行不确定 1列 输入
y = tf.placeholder(tf.float32, [None, 1])#行不确定 1列
#定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))#权值 1行10列 输入层1个神经元 中间层10个神经元
Biases_L1 = tf.Variable(tf.zeros([1, 10]))#10个偏置值
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + Biases_L1#信号总和
L1 = tf.nn.tanh(Wx_plus_b_L1)#中间层输出
#定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))#权值 10行1列 中间层10个神经元 输出层1个神经元
Biases_L2 = tf.Variable(tf.zeros([1, 1]))#1个偏置值
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + Biases_L2#信号总和
prediction = tf.nn.tanh(Wx_plus_b_L2)#输出 预测结果
#二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
#梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#变量初始化

    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})#训练 传入样本
    prediction_value = sess.run(prediction, feed_dict={x:x_data})#获得预测值

    #画图
    plt.figure("非线性回归")
    plt.scatter(x_data, y_data, color="blue", label="data")
    plt.plot(x_data, prediction_value, color="red", label="prediction", lw=3)
    plt.legend()
    plt.show()