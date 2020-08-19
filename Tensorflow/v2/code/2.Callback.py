import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
X_test_scaler = scaler.transform(X_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28, 28], name="flatten"))
model.add(tf.keras.layers.Dense(100, activation="relu"), name="dense")
model.add(tf.keras.layers.Dense(10, activation="softmax", name="output"))
model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# 回调函数
callback = [
    # tensorboarrd
    tf.keras.callbacks.TensorBoard("../board"),  # 路径
    # 保存模型
    tf.keras.callbacks.ModelCheckpoint("../save/callback.h5",
                                       monitor="val_loss", save_best_only=True),  # 路径+文件名 val_loss有改进时才会保存模型
    # 早期停止
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                     patience=5, min_delta=1e-3)  # val_loss变量5轮训练变化小于0.001则停止训练
]

model.fit(X_train_scaler, y_train, batch_size=500, epochs=10, validation_split=0.2, callbacks=callback)
