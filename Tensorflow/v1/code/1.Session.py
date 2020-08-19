import tensorflow as tf


# 初始化
mat1 = tf.constant([[3, 3]])  # 常量 1*2
mat2 = tf.constant([[2], [3]])  # 常量 2*1
mat3 = tf.ones([2, 5])  # 全0. 2*5
mat4 = tf.ones([3, 3])  # 全1. 3*3
mat5 = tf.fill([3, 3], 10.)  # 全10 3*3
mat6 = tf.random_normal([2, 2], mean=0.0, stddev=1.0)  # 随机值 2*2 均值0 标准差1

# 矩阵操作
mul = tf.matmul(mat1, mat2)  # 矩阵相乘
add = tf.add(mat4, mat5)
print(mul)
print(add)

# 形状
# 静态改变本身 动态返回新数组
a = tf.placeholder(tf.float32, [None, 2])  # [?,2]
print(a)
a.set_shape([3, 2])  # 静态变形
print(a)
# aa = a.set_shape([2, 3])  # 确定形状后不能再次修改
b = tf.placeholder(tf.float32, [5, 2])  # [5,2]
b_reshape = tf.reshape(b, [1, 2, 5])  # 动态变形可以改变维度 b形状不变
print(b)
print(b_reshape)

# 类型转换
a = tf.zeros([2, 2])
a_cast = tf.cast(a, tf.int32)
print(a)
print(a_cast)

with tf.Session() as sess:  # 定义会话 启动默认图
    print(sess.run(mul))  # 调用sess的run方法->矩阵相乘->生成两个常量
    print(sess.run(add))
    print(mat6.eval())  # 用于测试
