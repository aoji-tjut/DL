import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

max_time = 10
input_size = 1
lstm_size = 100

x = tf.placeholder(tf.float32, [None, max_time * input_size])
y = tf.placeholder(tf.float32, [None, max_time * input_size])

inputs = tf.reshape(x, [-1, max_time, input_size])
rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
outputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs, time_major=False, dtype=tf.float32)

w = tf.Variable(tf.zeros([lstm_size, max_time]) + 0.1)
b = tf.Variable(tf.zeros([max_time]))
y_predict = tf.matmul(state[1], w) + b

loss = tf.reduce_mean(tf.square(y_predict - y))
train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

plt.figure("RNN sin->cos", (12, 7))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
plt.ion()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(200):
        start, end = step * np.pi, (step + 1) * np.pi
        range = np.linspace(start, end, max_time)
        batch_x = np.sin(range).reshape(-1, max_time * input_size)
        batch_y = np.cos(range).reshape(-1, max_time * input_size)

        _, _loss, _y_predict = sess.run([train_op, loss, y_predict], feed_dict={x: batch_x, y: batch_y})
        print(_loss)

        plt.sca(ax1)
        plt.title("y")
        plt.plot(range, batch_y.reshape(-1), c='r')
        plt.ylim((-1.5, 1.5))
        plt.xlim(end - 6 * np.pi, end)
        plt.sca(ax2)
        plt.title("y_predict")
        plt.plot(range, _y_predict.reshape(-1), c='b')
        plt.ylim((-1.5, 1.5))
        plt.xlim(end - 6 * np.pi, end)
        plt.pause(0.1)

plt.ioff()
plt.show()
