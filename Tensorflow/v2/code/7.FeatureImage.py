import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 225.0
X_test = X_test.reshape(-1, 28, 28, 1) / 225.0

input = tf.keras.layers.Input(shape=[28, 28, 1])
conv = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, padding="same")(input)
bn = tf.keras.layers.BatchNormalization()(conv)
relu = tf.keras.layers.ReLU()(bn)
pool = tf.keras.layers.MaxPool2D()(relu)
flatten = tf.keras.layers.Flatten()(pool)
output = tf.keras.layers.Dense(10, activation="softmax")(flatten)

model = tf.keras.Model(inputs=input, outputs=output)
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))

# 特征图
layers = [conv, bn, relu, pool]
num = np.random.randint(0, 10000)
flag = 0

fig = plt.figure("conv/bn/relu/pool")
for layer in layers:
    model = tf.keras.Model(inputs=input, outputs=layer)
    output_shape = model.get_output_shape_at(0)
    image = model.predict(X_test[num].reshape(1, 28, 28, 1))
    image = image.reshape(output_shape[1:])

    for channel in range(output_shape[-1]):
        ax = fig.add_subplot(len(layers), output_shape[-1], channel + flag * output_shape[-1] + 1)
        plt.imshow(image[:, :, channel], cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    flag += 1

# 原图
plt.figure("Image")
plt.imshow(X_test[num].reshape(28, 28), cmap="gray")
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.show()
