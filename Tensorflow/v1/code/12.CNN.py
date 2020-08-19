import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


# 初始化权值
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


# 初始化偏置值
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# 卷积层
def conv2d_same(x, W):
    # input：[批次，高度，宽度，通道]
    # filter：[高度，宽度，输入通道，输出通道]
    # strides：[1，x步长，y步长，1]
    # padding：SAME补零，VALID不补零
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def conv2d_valid(x, W):
    # input：[批次，高度，宽度，通道]
    # filter：[高度，宽度，输入通道，输出通道]
    # strides：[1，x步长，y步长，1]
    # padding：SAME补零，VALID不补零
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")


# 池化层
def max_pool_2x2(x):
    # value：[批次，高度，宽度，通道]
    # ksize：[1，高度，宽度，1]
    # strides：[1，x步长，y步长，1]
    # padding：SAME可能补零，VALID不补零
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")


mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])  # 28*28
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 改变x形状
x_image = tf.reshape(x, [-1, 28, 28, 1])  # [批次，高度，宽度，通道]

# 第一个卷积层+池化层
W_conv1 = weight_variable([5, 5, 1, 64])  # 5*5的采样窗口，输入通道1，输出通道64
b_conv1 = bias_variable([64])  # 每个通道一个偏置值
h_conv1 = tf.nn.relu(conv2d_valid(x_image, W_conv1) + b_conv1)  # [?,28,28,1]->[?,24,24,64]
h_pool1 = max_pool_2x2(h_conv1)  # [?,24,24,64]->[?,12,12,64]

# 第二个卷积层+池化层
W_conv2 = weight_variable([3, 3, 64, 128])  # 3*3的采样窗口，输入通道64，输出通道128
b_conv2 = bias_variable([128])  # 每个通道一个偏置值
h_conv2 = tf.nn.relu(conv2d_same(h_pool1, W_conv2) + b_conv2)  # [?,12,12,64]->[?,12,12,128]
h_pool2 = max_pool_2x2(h_conv2)  # [?,12,12,128]->[?,6,6,128]

# 第一个全连接层
h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 6 * 128])  # 扁平化 把四维[?,6,6,64]转化为二维[?,6*6*64]
W_fc1 = weight_variable([6 * 6 * 128, 1024])  # 输入6*6*128个神经元 输出1024个神经元
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # [?,1024] = [?,6*6*128]*[6*6*128,1024] + [1024]
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第二个全连接层
W_fc2 = weight_variable([1024, 300])  # 输入1024个神经元 输出300个神经元
b_fc2 = bias_variable([300])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # [?,300] = [?,1024]*[1024,300] + [300]
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# 第三个全连接层
W_fc3 = weight_variable([300, 10])  # 十个分类
b_fc3 = bias_variable([10])
prediction = tf.matmul(h_fc2_drop, W_fc3) + b_fc3  # [?,10] = [?,300]*[300,10] + [10]

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct = (tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
saver = tf.train.Saver()

# 画图参数
plt.figure("CNN")
plt.xticks(np.linspace(0, 20, 21, dtype=np.int))
plt.axis([0, 21, 0.9, 1.0])
plt.ion()

# 训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.6})
        acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        plt.scatter(i + 1, acc_test)
        plt.pause(0.001)
        print("第%d次训练，集准确率%.2f%%" % (i + 1, (acc_test * 100)))
    saver.save(sess, "../save/cnn/cnn.ckpt")

# 画图
plt.ioff()
plt.show()
