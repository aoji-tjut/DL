import tensorflow as tf

x = tf.Variable([1, 2])  # 变量
a = tf.constant([3, 3])  # 常量
sub = tf.subtract(x, a)  # 减法
add = tf.add(sub, x)  # 加法
init = tf.global_variables_initializer()  # 变量初始化
with tf.Session() as sess:  # 定义会话(自动关闭)
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

state = tf.Variable(0)
one = tf.constant(1)
update = tf.assign_add(state, one)
# new_value = tf.add(state, one)  # 变量+常量=变量
# update = tf.assign(state, new_value)  # 赋值 state += new_value
init = tf.global_variables_initializer()  # 变量初始化
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))  # 输出state
    for i in range(5):
        sess.run(update)  # 赋值state=state+new_value <- new_value=state+1
        print(sess.run(state))  # 输出state
