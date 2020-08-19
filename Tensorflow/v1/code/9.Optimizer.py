import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

learning_rate = 0.003
batch_size = 32

x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise


class Net:
    def __init__(self, opt, **kwargs):
        self.x = tf.placeholder(tf.float32, [None, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])
        l = tf.layers.dense(self.x, 20, tf.nn.relu)
        out = tf.layers.dense(l, 1)
        self.loss = tf.losses.mean_squared_error(self.y, out)
        self.train = opt(learning_rate, **kwargs).minimize(self.loss)


net_SGD = Net(tf.train.GradientDescentOptimizer)
net_Momentum = Net(tf.train.MomentumOptimizer, momentum=0.9)
net_RMSprop = Net(tf.train.RMSPropOptimizer)
net_Adam = Net(tf.train.AdamOptimizer)
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

losses_his = [[], [], [], []]

# 训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(500):
        # 随机取batch_size个样本训练
        index = np.random.randint(0, x.shape[0], batch_size)
        batch_x = x[index]
        batch_y = y[index]

        # 4个优化器与对应损失打包 循环四次
        for net, l_his in zip(nets, losses_his):
            _, loss = sess.run([net.train, net.loss], {net.x: batch_x, net.y: batch_y})
            l_his.append(loss)

# 画图
plt.figure("Optimizer")
labels = ["SGD", "Momentum", "RMSprop", "Adam"]
for i, l_his in enumerate(losses_his):  # 循环四次
    plt.plot(l_his, label=labels[i])  # 横坐标缺省
plt.legend()
plt.xlabel('Step')
plt.ylabel('Loss')
plt.ylim((0, 0.3))
plt.show()
