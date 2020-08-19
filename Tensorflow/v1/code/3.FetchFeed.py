import tensorflow as tf

# fetch
a = tf.constant(1.0)
b = tf.constant(3.0)
c = tf.constant(5.0)
add = tf.add(a, b)
mul = tf.multiply(c, add)
with tf.Session() as sess:
    result = sess.run([mul, add])
    print(result)

# feed run的时候传入
a = tf.placeholder(tf.float32)  # 创建占位符
b = tf.placeholder(tf.float32)  # 创建占位符
output = tf.multiply(a, b)
with tf.Session() as sess:
    print(sess.run(output, feed_dict={a: 5., b: 10.}))

