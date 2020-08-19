import tensorflow as tf
import numpy as np

# 常量
t = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
print(t)
print(t - 1)
print(t[:, 1:])
print(t[..., 0])  # 向量
print(tf.transpose(t))  # 转置
print(np.array(t))  # 转换成numpy
print(tf.constant(2))  # 0维
print()

# 字符串
str = tf.constant(["q", "qw", "qwe", "qwer"])
print(str)
print(tf.strings.length(str))
print(tf.strings.unicode_decode(str, "UTF8"))  # 转换编码
print()

# 变量
v = tf.Variable([[1, 2, 3], [4, 5, 6]])
print(v)  # tensor+array
print(v.value())  # tensor
print(v.numpy())  # array
v.assign(2 * v)  # 乘2
print(v.numpy())
v[0, 1].assign(666)  # 改某个值
print(v.numpy())
v[1].assign([666, 666, 666])  # 改某行值
print(v.numpy())
print()

# 其他
a = tf.fill([2, 3], 5.0)  # 全填充
print(a)
a = tf.cast(a, tf.int32)  # 类型转换
print(a)
a = tf.zeros_like(a)  # 形状同a的全0矩阵
print(a)
a = tf.random.normal([2, 2], mean=0, stddev=1)  # 正态分布随机数 均值0方差1
print(a)
a = tf.random.uniform([2, 2], maxval=10)  # [0,10)均匀分布随机数
print(a)
a = tf.range(24)
print(a)
print()

# 维度
a = tf.reshape(a, [1, 2, 3, 4])  # 改变形状
print(a)
a = tf.expand_dims(a, axis=3)  # 在axis=3处增加维度
print(a)
a = tf.squeeze(a, axis=3)  # 删除axis=3的维度 只能删除维度为1的轴
print(a)
a = tf.transpose(a, [0, 1, 3, 2])  # 交换2、3两个维度
print(a)
print()

# 计算
a = tf.constant([2., 3.])
b = tf.constant([4., 8.])
print(a)
print(b)
c = b / a  # 除
print(c)
c = b // a  # 整除
print(c)
c = b % a  # 余
print(c)
c = tf.square(a)  # 平方
print(c)
c = tf.pow(a, 3)  # 3次方
print(c)
c = tf.sqrt(b)  # 开平方
print(c)
c = tf.pow(b, 1 / 3)  # 开3次方
print(c)
c = tf.exp(a)  # e^a
print(c)
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([1, 2, 3, 4, 5, 6], shape=[3, 2])
c = tf.matmul(a, b)  # 矩阵乘法 [2,3] * [3,2] = [2,2]
print(a)
print(b)
print(c)
print()

# 合并分割
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
print(a)
print(b)
c = tf.stack([a, b], axis=0)  # 在第0个维度创建个数维度
print(c)
c = tf.concat([a, b], axis=0)  # 纵向拼接
print(c)
c = tf.concat([a, b], axis=1)  # 横向拼接
print(c)
c1, c2 = tf.split(c, 2, axis=0)  # 按第0个维度平均分割成2分
print(c1, c2, sep='\n')
c1, c2, c3 = tf.split(c, [1, 2, 3], axis=1)  # 按第1个维度按list分割成3分
print(c1, c2, c3, sep='\n')
print()

# One-Hot编码
a = tf.constant([0, 1, 2, 3, 4, 5])
print(a)
a = tf.one_hot(a, depth=6)  # 每个编码长度为6
print(a)
