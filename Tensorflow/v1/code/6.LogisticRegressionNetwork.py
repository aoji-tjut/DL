import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)  # 独热码
# 每个批次大小
batch_size = 100  # 一次放入一个批次的图片 即100个
# 计算批次总数
n_batch = mnist.train.num_examples // batch_size
# placeholder
x = tf.placeholder(tf.float32, [None, 784])  # 图像 28*28=784个像素
y = tf.placeholder(tf.float32, [None, 10])  # 标签 0~9共10个数字
# 输出层
W = tf.Variable(tf.zeros([784, 10]))  # 权值
b = tf.Variable(tf.zeros([10]))  # 偏置值
prediction = tf.nn.softmax(tf.matmul(x, W) + b)  # [?,10] = [?,784]*[784,10] + [10]
# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 返回bool列表求预测准确度
correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # axis=1 返回一行最大值索引 比较索引是否相同
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # 把bool转化为float 对1、0求平均值得到准确率

# 画图参数
plt.figure("手写数字识别准确率")
plt.axis([0, 21, 0.8, 0.95])
plt.xticks(np.linspace(0, 20, 21, dtype=int))
plt.ion()

# 会话
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):  # 把所有图片训练20次
        for batch in range(n_batch):  # 一共训练n_batch个批次将所有图片训练完毕
            batch_x, batch_y = mnist.train.next_batch(batch_size)  # 一次获得batch_size个数据
            sess.run(train, feed_dict={x: batch_x, y: batch_y})  # 训练
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        plt.scatter(i + 1, acc)
        plt.pause(0.001)
        print("第%d次测试，准确率%.2f%%" % (i + 1, (acc * 100)))

plt.ioff()
plt.show()
