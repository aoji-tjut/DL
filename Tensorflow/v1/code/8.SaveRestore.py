import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)
loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
init = tf.global_variables_initializer()
correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
saver = tf.train.Saver()#存储器

# save
with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("第%d次测试，准确率%.2f%%" % (i + 1, (acc * 100)))
    saver.save(sess, "../save/bp/bp.ckpt")  # save

# restore
with tf.Session() as sess:
    sess.run(init)
    print("加载模型前准确率：%.2f%%\n" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    if os.path.exists("../save/bp/checkpoint"):
        saver.restore(sess, "../save/bp/bp.ckpt")  # restore 可以从上次保留的模型继续训练 不需要tf.global_variables_initializer()
    print("加载模型后准确率：%.2f%%\n" % (sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}) * 100))
