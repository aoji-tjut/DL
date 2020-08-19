import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

input_size = 28  # 每行输入28个像素
max_time = 28  # 输入28次
lstm_size = 100  # 100个神经元
n_classes = 10  # 10个分类0～9
batch_size = 50
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))  # [100,10]
biases = tf.Variable(tf.constant(0., shape=[n_classes]))  # [10]

# ----------------------------------------------------------------------------------------------------------------------
# 输入层
inputs = tf.reshape(x, [-1, max_time, input_size])  # [?,28,28]

# 隐藏层ddddd
cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0)  # 神经元个数 遗忘门系数 1不忘记 0全忘记
outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, time_major=False)
# time_major=False:[batch_size, max_time, ...]    time_major=True:[max_time, batch_size, ...]
# outputs = [batch_size, max_time, cell.output_size] = [?,28,100] = 每个时间步输出的结果
# state = [batch_size, cell.state_size] = [?,100]    state[0] = ct = 长期记忆信息    state[1] = ht = 短期记忆信息
# state短期记忆信息 = outputs最后一个时间步输出 = ht    即state[1] = outputs[:,-1,:]

# 输出层
prediction = tf.matmul(state[1], weights) + biases  # [?,10] = [?,100]*[100,10] + [10]
# ----------------------------------------------------------------------------------------------------------------------

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
saver = tf.train.Saver()

plt.figure("RNN")
plt.xticks(np.linspace(0, 20, 21, dtype=np.int))
plt.axis([0, 21, 0.9, 1.0])
plt.ion()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        plt.scatter(i + 1, acc_test)
        plt.pause(0.001)
        print("第%d次训练，准确率%.2f%%" % (i + 1, (acc_test * 100)))
    saver.save(sess, "../save/rnn/rnn.ckpt")

plt.ioff()
plt.show()
