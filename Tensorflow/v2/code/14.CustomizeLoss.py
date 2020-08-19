import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation="relu", input_shape=[8]))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(1))
model.summary()


def Loss(y_true, y_predict):  # true在前 predict在后
    return tf.reduce_mean(tf.square(y_true - y_predict))


model.compile(loss=Loss, optimizer="sgd", metrics=["mse", "acc"])
model.fit(X_train_scaler, y_train, batch_size=500, epochs=10, validation_split=0.2)

model.evaluate(X_test_scaler, y_test)
