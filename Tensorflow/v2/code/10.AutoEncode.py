import tensorflow as tf
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# encoder
encode_input = tf.keras.layers.Input(shape=[784])
a = tf.keras.layers.Dense(512, activation="relu")(encode_input)
a = tf.keras.layers.Dense(256, activation="relu")(a)
a = tf.keras.layers.Dense(128, activation="relu")(a)
a = tf.keras.layers.Dense(64, activation="relu")(a)
a = tf.keras.layers.Dense(32, activation="relu")(a)
a = tf.keras.layers.Dense(16, activation="relu")(a)
a = tf.keras.layers.Dense(8, activation="relu")(a)
encode_output = tf.keras.layers.Dense(4, activation="relu")(a)

# decoder
a = tf.keras.layers.Dense(4, activation="relu")(encode_output)
a = tf.keras.layers.Dense(8, activation="relu")(a)
a = tf.keras.layers.Dense(16, activation="relu")(a)
a = tf.keras.layers.Dense(32, activation="relu")(a)
a = tf.keras.layers.Dense(64, activation="relu")(a)
a = tf.keras.layers.Dense(128, activation="relu")(a)
a = tf.keras.layers.Dense(256, activation="relu")(a)
a = tf.keras.layers.Dense(512, activation="relu")(a)
decode_output = tf.keras.layers.Dense(784, activation="relu")(a)

# 训练encode-decode模型
encode_decode = tf.keras.Model(inputs=encode_input, outputs=decode_output)
encode_decode.summary()
encode_decode.compile(optimizer="adam", loss="mse")
history = encode_decode.fit(X_train, X_train, epochs=5, validation_data=(X_test, X_test))

# 绘制loss
plt.figure("Loss")
plt.plot(history.epoch, history.history["loss"], label="loss")
plt.plot(history.epoch, history.history["val_loss"], label="val_loss")
plt.legend()

# 原图
fig = plt.figure("AE")
for i in range(10):
    ax = fig.add_subplot(3, 10, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# 只获取encode模型
encode = tf.keras.Model(inputs=encode_input, outputs=encode_output)
y_encode = encode.predict(X_test[:10])
for i in range(10):
    ax = fig.add_subplot(3, 10, i + 1 + 10)
    plt.imshow(y_encode[i].reshape(2, 2), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# 获取encode-decode模型
y_encode_decode = encode_decode.predict(X_test[:10])
for i in range(10):
    ax = fig.add_subplot(3, 10, i + 1 + 20)
    plt.imshow(y_encode_decode[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

'''
# 卷积编码
encode_input = tf.keras.Input(shape=(28, 28, 1))
h1 = tf.keras.layers.Conv2D(16, 3, activation="relu")(encode_input)
h1 = tf.keras.layers.Conv2D(32, 3, activation="relu")(h1)
h1 = tf.keras.layers.MaxPool2D()(h1)
h1 = tf.keras.layers.Conv2D(32, 3, activation="relu")(h1)
h1 = tf.keras.layers.Conv2D(16, 3, activation="relu")(h1)  # [8,8,16]
encode_output = tf.keras.layers.GlobalMaxPool2D()(h1)  # [-1,16]

# 反卷积解码
h2 = tf.keras.layers.Reshape((4, 4, 1))(encode_output)  # [-1,16]->[-1,4,4,1]
h2 = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu")(h2)
h2 = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu")(h2)
h2 = tf.keras.layers.UpSampling2D(3)(h2)
h2 = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu")(h2)
decode_output = tf.keras.layers.Conv2DTranspose(1, 3, activation="relu")(h2)

# 自动编码器
auto_encoder_model = tf.keras.models.Model(inputs=encode_input, outputs=decode_output)
auto_encoder_model.summary()
'''
