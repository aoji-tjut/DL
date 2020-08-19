import tensorflow as  tf


def fun(x1, x2):
    return 4 * (x1 ** 3) * x2 + 3 * x1 * (x2 ** 2) + 2 * x1 + x2 + 10


# 多个变量
v1 = tf.Variable(2.0)
v2 = tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = fun(v1, v2)
dz_v1, dz_v2 = tape.gradient(z, [v1, v2])
print(dz_v1)
print(dz_v2)

c1 = tf.constant(2.0)
c2 = tf.constant(3.0)
with tf.GradientTape() as tape:
    # 常量需要watch
    tape.watch(c1)
    tape.watch(c2)
    z = fun(c1, c2)
dz_c1, dz_c2 = tape.gradient(z, [c1, c2])
print(dz_c1)
print(dz_c2)

# 多个函数
v = tf.Variable(2.0)
with tf.GradientTape() as tape:
    z1 = 2 * v ** 2
    z2 = 4 * v ** 4
dz_v = tape.gradient([z1, z2], v)
print(dz_v)

# 二阶导数
v1 = tf.Variable(2.0)
v2 = tf.Variable(3.0)
# 多次使用persistent=True
with tf.GradientTape(persistent=True) as out_tape:
    with tf.GradientTape(persistent=True) as in_tape:
        z = fun(v1, v2)
    in_grad = in_tape.gradient(z, [v1, v2])
out_grad = [out_tape.gradient(i, [v1, v2]) for i in in_grad]
print(out_grad[0][0])  # z对x1二阶导
print(out_grad[0][1])  # z对x2对x1
print(out_grad[1][0])  # z对x1对x2
print(out_grad[1][1])  # z对x2二阶导
del out_tape, in_tape
