import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

# 每个批次大小
batch_size = 100
# 计算批次总数
n_batch = mnist.train.num_examples // batch_size
# placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)  # 工作神经元百分比
# 隐藏层
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))  # 权值 标准差0.1
b1 = tf.Variable(tf.zeros([500]) + 0.1)
L1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # [?,500] = [?,784]*[784*500] + [500]
L1_drop = tf.nn.dropout(L1, keep_prob)  # 设置工作神经元比例
W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300]) + 0.1)
L2 = tf.nn.relu(tf.matmul(L1_drop, W2) + b2)  # [?,300] = [?,500]*[500*300] + [300]
L2_drop = tf.nn.dropout(L2, keep_prob)
W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]) + 0.1)
prediction = tf.matmul(L2_drop, W3) + b3  # [?,10] = [?,300]*[300*10] + [10] 使用交叉熵损失要传入线性输出 所以不需要激活函数
# 损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))  # 交叉熵
# 训练
train = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)  # adam优化器
# 返回bool列表求预测准确度
correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
# 保存
saver = tf.train.Saver()

# 画图参数
plt.figure("DNN")
plt.xticks(np.linspace(0, 20, 21, dtype=np.int))
plt.axis([0, 21, 0.9, 1.0])
plt.ion()

# 会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.6})  # 用60%的神经元训练 防止过拟合
        acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        plt.scatter(i + 1, acc_test)
        plt.pause(0.001)
        print("第%d次训练，准确率%.2f%%" % (i + 1, (acc_test * 100)))
    saver.save(sess, "../save/dnn/dnn.ckpt")

plt.ioff()
plt.show()
