import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# [60000,28,28]  [60000,]  [10000,28,28]  [10000,]
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding="valid",
                                 activation="relu", input_shape=[28, 28, 1], name="conv1"))  # [-1,28,28,1]
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="pool1"))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same",
                                 activation="relu", name="conv2"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="pool2"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation="selu", name="dense1"))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(256, activation="selu", name="dense2"))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(10, activation="softmax", name="output"))
model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
callback = [tf.keras.callbacks.TensorBoard("../board")]
history = model.fit(X_train, y_train, batch_size=500, epochs=10, validation_data=(X_test, y_test), callbacks=callback)

model.save("../save/cnn.h5")

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
