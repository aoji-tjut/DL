import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing.data
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)


def build_model(hidden_layers=1, layer_size=30, learning_rate=0.001):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(layer_size, activation="relu", input_shape=[8]))  # [-1,8]
    for _ in range(hidden_layers - 1):
        model.add(tf.keras.layers.Dense(layer_size, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model


sklearn_model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_model)
sklearn_model.fit(X_train_scaler, y_train, epochs=10, validation_data=(X_test_scaler, y_test))

param = {
    "hidden_layers": np.linspace(1, 10, 10, dtype=np.int),
    "layer_size": np.linspace(10, 100, 10, dtype=np.int),
    "learning_rate": np.linspace(0.0001, 0.1, 300, dtype=np.float)
}

random_search_cv = RandomizedSearchCV(sklearn_model, param, n_jobs=-1)
random_search_cv.fit(X_train_scaler, y_train, epochs=10, validation_data=(X_test_scaler, y_test))

print(random_search_cv.best_params_)
print(random_search_cv.best_score_)

# pd.DataFrame(history.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.show()
