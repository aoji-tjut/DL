import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
X_test_scaler = scaler.transform(X_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))  # [-1,28,28]
model.add(tf.keras.layers.Dense(512, activation="relu", name="dense1"))  # [-1,784]
model.add(tf.keras.layers.Dense(256, activation="relu", name="dense2"))
model.add(tf.keras.layers.Dense(128, activation="relu", name="dense3"))
model.add(tf.keras.layers.Dense(64, activation="relu", name="dense4"))
model.add(tf.keras.layers.Dense(10, activation="softmax", name="output"))
model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
callback = [tf.keras.callbacks.TensorBoard("../board")]
history = model.fit(X_train_scaler, y_train, batch_size=500, epochs=20, validation_split=0.2, callbacks=callback)

model.save("../save/dnn.h5")

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
