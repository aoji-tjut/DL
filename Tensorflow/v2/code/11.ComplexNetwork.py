import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing.data  # [20640,8]
y = housing.target  # [20640,]

# [15480,8]  [5160,8]  [15480,]  [5160,]
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

X_train_scaler_wide = X_train_scaler[:, :5]
X_train_scaler_deep = X_train_scaler[:, -6:]
X_test_scaler_wide = X_test_scaler[:, :5]
X_test_scaler_deep = X_test_scaler[:, -6:]

# 多输入
input_wide = tf.keras.layers.Input(shape=[5], name="input_wide")
input_deep = tf.keras.layers.Input(shape=[6], name="input_deep")
h1 = tf.keras.layers.Dense(30, activation="relu", name="hidden1")(input_deep)
h2 = tf.keras.layers.Dense(30, activation="relu", name="hidden2")(h1)
concat = tf.keras.layers.concatenate([input_wide, h2])
output = tf.keras.layers.Dense(1, name="output")(concat)
model = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=output)
model.summary()

model.compile(loss="mean_squared_error", optimizer="sgd")
callback = [tf.keras.callbacks.TensorBoard("../board")]
history = model.fit([X_train_scaler_wide, X_train_scaler_deep], y_train, epochs=10, callbacks=callback)

model.evaluate([X_test_scaler_wide, X_test_scaler_deep], y_test)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# 多输出
input_wide = tf.keras.layers.Input(shape=[5], name="input_wide")
input_deep = tf.keras.layers.Input(shape=[6], name="input_deep")
h1 = tf.keras.layers.Dense(30, activation="relu", name="hidden1")(input_deep)
h2 = tf.keras.layers.Dense(30, activation="relu", name="hidden2")(h1)
concat = tf.keras.layers.concatenate([input_wide, h2])
output1 = tf.keras.layers.Dense(1, name="output1")(concat)
output2 = tf.keras.layers.Dense(1, name="output2")(h2)
model = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output1, output2])
model.summary()

model.compile(loss="mean_squared_error", optimizer="sgd")
callback = [tf.keras.callbacks.TensorBoard("../board")]
history = model.fit([X_train_scaler_wide, X_train_scaler_deep], [y_train, y_train], epochs=10, callbacks=callback)

model.evaluate([X_test_scaler_wide, X_test_scaler_deep], [y_test, y_test])

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
