import tensorflow as tf

train_dir = "../data/10-monkey-species/training"
valid_dir = "../data/10-monkey-species/validation"

height = 128
width = 128
channels = 3
batch_size = 64
num_classes = 10

# 图片增强生成器
train_image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,  # 归一化
    rotation_range=40,  # 随机旋转 -40～40度
    width_shift_range=0.2,  # 横向随机位移 -20%~20%
    height_shift_range=0.2,  # 纵向随机位移 -20%~20%
    shear_range=0.2,  # 错切变换 -20%~20%
    zoom_range=0.2,  # 随机缩放 -20%~20%
    horizontal_flip=True,  # 随机水平翻转
    fill_mode="nearest"  # 放大操作时填充像素方法
)

# 增强后的图片
train_generator = train_image_data_generator.flow_from_directory(
    train_dir,  # 路径
    target_size=(height, width),  # 目标尺寸
    batch_size=batch_size,  # 每批次图片个数
    seed=666,  # 随机数
    shuffle=True,  # 乱序
    class_mode="categorical"  # One-Hot编码标签
)

# 图片增强生成器
valid_image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

# 增强后的图片
valid_generator = valid_image_data_generator.flow_from_directory(
    valid_dir,
    target_size=(height, width),
    batch_size=batch_size,
    seed=666,
    shuffle=False,
    class_mode="categorical"
)

print(train_generator.samples)
print(valid_generator.samples)

x, y = train_generator.next()
print(x.shape, y.shape)
print(y)

# 训练
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu",
                                 input_shape=[height, width, channels]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="selu"))
model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
# 使用fit_generator
history = model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=10,
                              validation_data=valid_generator, validation_steps=valid_generator.samples // batch_size)
