import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# gpu设置
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices("GPU")

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
X_test_scaler = scaler.transform(X_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

#分布式
strategy=tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))  # [-1,28,28]
    model.add(tf.keras.layers.Dense(512, activation="relu", name="dense1"))  # [-1,784]
    model.add(tf.keras.layers.Dense(256, activation="relu", name="dense2"))
    model.add(tf.keras.layers.Dense(128, activation="relu", name="dense3"))
    model.add(tf.keras.layers.Dense(64, activation="relu", name="dense4"))
    model.add(tf.keras.layers.Dense(10, activation="softmax", name="output"))
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train_scaler, y_train, batch_size=500, epochs=20, validation_split=0.2)
