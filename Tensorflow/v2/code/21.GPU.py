import tensorflow as tf

# 记录变量在gpu的位置
tf.debugging.set_log_device_placement(True)
# 自动指定处理设备
tf.config.set_soft_device_placement(True)

# 物理gpu
gpus = tf.config.experimental.list_physical_devices("GPU")
print("物理gpu:", len(gpus))
for gpu in gpus:
    # 设置内存自增长
    tf.config.experimental.set_memory_growth(gpu, True)
if len(gpus) > 0:
    # 设置第0个gpu可见
    tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    # 切分第0个gpu 逻辑分区2个 每个1G
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
    )

# 逻辑gpu 个数由物理gpu决定
logical_gpus = tf.config.experimental.list_logical_devices("GPU")
print("逻辑gpu:", len(logical_gpus))
c = []
if len(logical_gpus) > 0:
    for gpu in logical_gpus:
        print(gpu.name)
        # 一层网络可以串行使用多个gpu运算
        with tf.device(gpu.name):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c.append(tf.matmul(a, b))

# cpu
with tf.device("/CPU:0"):
    sum = tf.add_n(c)
print(sum)
