import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 参数摘要
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)  # 计算平均值 后面的计算都要用到
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 收集参数
        tf.summary.scalar("mean", mean)  # 平均值
        tf.summary.scalar("stddev", stddev)  # 标准差
        tf.summary.scalar("max", tf.reduce_max(var))  # 最大值
        tf.summary.scalar("min", tf.reduce_min(var))  # 最小值
        tf.summary.histogram("histogram", var)  # 直方图


# 载入数据集
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

# 每个批次大小
batch_size = 100  # 一小批100个

# 计算批次总数
n_batch = mnist.train.num_examples // batch_size

# placeholder
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="x_input")
    y = tf.placeholder(tf.float32, [None, 10], name="y_input")

# 网络层
with tf.name_scope("layer"):
    with tf.name_scope("Weight"):
        W = tf.Variable(tf.zeros([784, 10]))
        variable_summaries(W)
    with tf.name_scope("biases"):
        b = tf.Variable(tf.zeros([10]))
        variable_summaries(b)
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope("softmax"):
        prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 损失
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(y - prediction))
    tf.summary.scalar("loss", loss)

# 训练
with tf.name_scope("train"):
    train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 准确率
with tf.name_scope("acc"):
    # 返回bool列表求预测准确度
    with tf.name_scope("correct"):
        correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    # 求准确率
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

# 初始化变量
init = tf.global_variables_initializer()

# 合并summary 收集scalar
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("../board", sess.graph)  # 写入图

    for i in range(10):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train], feed_dict={x: batch_x, y: batch_y})  # 合并写可以提高速度
        writer.add_summary(summary, i)  # 每一次完整的训练写入tensorboard 写入变量
        # tensorboard --logdir="/Users/aoji/Documents/PyCharmProject/ML/Tensorflow/board" # board文件夹绝对路径
