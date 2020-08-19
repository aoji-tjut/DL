import tensorflow as tf

input = tf.keras.layers.Input(shape=[224, 224, 3])

# conv1
conv = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same")(input)
relu = tf.keras.layers.ReLU()(conv)
pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(relu)
bn = tf.keras.layers.BatchNormalization()(pool)

# conv2
conv = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same")(bn)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding="same")(relu)
relu = tf.keras.layers.ReLU()(conv)
bn = tf.keras.layers.BatchNormalization()(relu)

# conv3a
inception1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(bn)
# 1
conv = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same")(inception1)
_1 = tf.keras.layers.ReLU()(conv)
# 2
conv = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding="same")(inception1)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(relu)
_2 = tf.keras.layers.ReLU()(conv)
# 3
conv = tf.keras.layers.Conv2D(filters=16, kernel_size=1, strides=1, padding="same")(inception1)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding="same")(relu)
_3 = tf.keras.layers.ReLU()(conv)
# 4
pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inception1)
conv = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same")(pool)
_4 = tf.keras.layers.ReLU()(conv)
# concatenate
concatenate = tf.keras.layers.concatenate([_1, _2, _3, _4])

# conv3b
inception2 = concatenate
# 1
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same")(inception2)
_1 = tf.keras.layers.ReLU()(conv)
# 2
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same")(inception2)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding="same")(relu)
_2 = tf.keras.layers.ReLU()(conv)
# 3
conv = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same")(inception2)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=96, kernel_size=5, strides=1, padding="same")(relu)
_3 = tf.keras.layers.ReLU()(conv)
# 4
pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inception2)
conv = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same")(pool)
_4 = tf.keras.layers.ReLU()(conv)
# concatenate
concatenate = tf.keras.layers.concatenate([_1, _2, _3, _4])

# conv4a
inception3 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(concatenate)
# 1
conv = tf.keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, padding="same")(inception3)
_1 = tf.keras.layers.ReLU()(conv)
# 2
conv = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding="same")(inception3)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=208, kernel_size=3, strides=1, padding="same")(relu)
_2 = tf.keras.layers.ReLU()(conv)
# 3
conv = tf.keras.layers.Conv2D(filters=16, kernel_size=1, strides=1, padding="same")(inception3)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=48, kernel_size=5, strides=1, padding="same")(relu)
_3 = tf.keras.layers.ReLU()(conv)
# 4
pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inception3)
conv = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same")(pool)
_4 = tf.keras.layers.ReLU()(conv)
# concatenate
concatenate = tf.keras.layers.concatenate([_1, _2, _3, _4])

# conv4b
inception4 = concatenate
# 1
conv = tf.keras.layers.Conv2D(filters=160, kernel_size=1, strides=1, padding="same")(inception4)
_1 = tf.keras.layers.ReLU()(conv)
# 2
conv = tf.keras.layers.Conv2D(filters=112, kernel_size=1, strides=1, padding="same")(inception4)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=224, kernel_size=3, strides=1, padding="same")(relu)
_2 = tf.keras.layers.ReLU()(conv)
# 3
conv = tf.keras.layers.Conv2D(filters=24, kernel_size=1, strides=1, padding="same")(inception4)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding="same")(relu)
_3 = tf.keras.layers.ReLU()(conv)
# 4
pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inception4)
conv = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same")(pool)
_4 = tf.keras.layers.ReLU()(conv)
# concatenate
concatenate = tf.keras.layers.concatenate([_1, _2, _3, _4])

# softmax1
pool = tf.keras.layers.AvgPool2D(pool_size=5, strides=3, padding="valid")(inception4)
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same")(pool)
relu = tf.keras.layers.ReLU()(conv)
flatten = tf.keras.layers.Flatten()(relu)
dense = tf.keras.layers.Dense(1024)(flatten)
relu = tf.keras.layers.ReLU()(dense)
dropout = tf.keras.layers.Dropout(0.7)(relu)
dense = tf.keras.layers.Dense(1024)(dropout)
relu = tf.keras.layers.ReLU()(dense)
dropout = tf.keras.layers.Dropout(0.7)(relu)
dense = tf.keras.layers.Dense(1000)(dropout)
softmax1 = tf.keras.layers.Softmax()(dense)

# conv4c
inception5 = concatenate
# 1
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same")(inception5)
_1 = tf.keras.layers.ReLU()(conv)
# 2
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same")(inception5)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(relu)
_2 = tf.keras.layers.ReLU()(conv)
# 3
conv = tf.keras.layers.Conv2D(filters=24, kernel_size=1, strides=1, padding="same")(inception5)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding="same")(relu)
_3 = tf.keras.layers.ReLU()(conv)
# 4
pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inception5)
conv = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same")(pool)
_4 = tf.keras.layers.ReLU()(conv)
# concatenate
concatenate = tf.keras.layers.concatenate([_1, _2, _3, _4])

# conv4d
inception6 = concatenate
# 1
conv = tf.keras.layers.Conv2D(filters=112, kernel_size=1, strides=1, padding="same")(inception6)
_1 = tf.keras.layers.ReLU()(conv)
# 2
conv = tf.keras.layers.Conv2D(filters=144, kernel_size=1, strides=1, padding="same")(inception6)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=288, kernel_size=3, strides=1, padding="same")(relu)
_2 = tf.keras.layers.ReLU()(conv)
# 3
conv = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same")(inception6)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding="same")(relu)
_3 = tf.keras.layers.ReLU()(conv)
# 4
pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inception6)
conv = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same")(pool)
_4 = tf.keras.layers.ReLU()(conv)
# concatenate
concatenate = tf.keras.layers.concatenate([_1, _2, _3, _4])

# conv4e
inception7 = concatenate
# 1
conv = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="same")(inception7)
_1 = tf.keras.layers.ReLU()(conv)
# 2
conv = tf.keras.layers.Conv2D(filters=160, kernel_size=1, strides=1, padding="same")(inception7)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=320, kernel_size=3, strides=1, padding="same")(relu)
_2 = tf.keras.layers.ReLU()(conv)
# 3
conv = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same")(inception7)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, padding="same")(relu)
_3 = tf.keras.layers.ReLU()(conv)
# 4
pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inception7)
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same")(pool)
_4 = tf.keras.layers.ReLU()(conv)
# concatenate
concatenate = tf.keras.layers.concatenate([_1, _2, _3, _4])

# softmax2
pool = tf.keras.layers.AvgPool2D(pool_size=5, strides=3, padding="valid")(inception7)
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same")(pool)
relu = tf.keras.layers.ReLU()(conv)
flatten = tf.keras.layers.Flatten()(relu)
dense = tf.keras.layers.Dense(1024)(flatten)
relu = tf.keras.layers.ReLU()(dense)
dropout = tf.keras.layers.Dropout(0.7)(relu)
dense = tf.keras.layers.Dense(1024)(dropout)
relu = tf.keras.layers.ReLU()(dense)
dropout = tf.keras.layers.Dropout(0.7)(relu)
dense = tf.keras.layers.Dense(1000)(dropout)
softmax2 = tf.keras.layers.Softmax()(dense)

# conv5a
inception8 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(concatenate)
# 1
conv = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="same")(inception8)
_1 = tf.keras.layers.ReLU()(conv)
# 2
conv = tf.keras.layers.Conv2D(filters=160, kernel_size=1, strides=1, padding="same")(inception8)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=320, kernel_size=3, strides=1, padding="same")(relu)
_2 = tf.keras.layers.ReLU()(conv)
# 3
conv = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same")(inception8)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, padding="same")(relu)
_3 = tf.keras.layers.ReLU()(conv)
# 4
pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inception8)
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same")(pool)
_4 = tf.keras.layers.ReLU()(conv)
# concatenate
concatenate = tf.keras.layers.concatenate([_1, _2, _3, _4])

# conv5b
inception9 = concatenate
# 1
conv = tf.keras.layers.Conv2D(filters=384, kernel_size=1, strides=1, padding="same")(inception9)
_1 = tf.keras.layers.ReLU()(conv)
# 2
conv = tf.keras.layers.Conv2D(filters=192, kernel_size=1, strides=1, padding="same")(inception9)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding="same")(relu)
_2 = tf.keras.layers.ReLU()(conv)
# 3
conv = tf.keras.layers.Conv2D(filters=48, kernel_size=1, strides=1, padding="same")(inception9)
relu = tf.keras.layers.ReLU()(conv)
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, padding="same")(relu)
_3 = tf.keras.layers.ReLU()(conv)
# 4
pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inception9)
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same")(pool)
_4 = tf.keras.layers.ReLU()(conv)
# concatenate
concatenate = tf.keras.layers.concatenate([_1, _2, _3, _4])

# softmax3
pool = tf.keras.layers.AvgPool2D(pool_size=7, strides=3, padding="valid")(concatenate)
flatten = tf.keras.layers.Flatten()(pool)
dense = tf.keras.layers.Dense(1024)(flatten)
relu = tf.keras.layers.ReLU()(dense)
dropout = tf.keras.layers.Dropout(0.4)(relu)
dense = tf.keras.layers.Dense(1000)(dropout)
softmax3 = tf.keras.layers.Softmax()(dense)

model = tf.keras.Model(inputs=input, outputs=softmax3)
model.summary()
