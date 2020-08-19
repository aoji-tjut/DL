import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=5, strides=1, activation="relu", input_shape=[32, 32, 1]))
model.add(tf.keras.layers.AveragePooling2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, activation="relu"))
model.add(tf.keras.layers.AveragePooling2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120, activation="relu"))
model.add(tf.keras.layers.Dense(84, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.summary()
