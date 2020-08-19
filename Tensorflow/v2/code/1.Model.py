import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pprint
from sklearn.preprocessing import StandardScaler

fashion_mnist = tf.keras.datasets.fashion_mnist
# [60000,28,28]  [60000,]  [10000,28,28]  [10000,]
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 归一化  [-1,28,28]->[-1,784]->transform->[-1,28,28]
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
X_test_scaler = scaler.transform(X_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

# 序列模型
# Conv/Dense->BN->Activation->Dropout
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))  # 第一层需要标明shape
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# 输出模型架构
model.summary()

# 编译
# One-Hot编码  loss="categorical_crossentropy"
# 具体数字编码  loss="sparse_categorical_crossentropy" 函数内部对y_test进行One-Hot编码实现
# 多类问题  output=类别  loss="categorical_crossentropy"/"sparse_categorical_crossentropy"
# 二类问题  output=1  loss="binary_crossentropy"
# 回归问题  output=1  loss="mean_squared_error"
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# 拟合
history = model.fit(X_train_scaler, y_train, batch_size=500, epochs=10, validation_split=0.2)
print(history.history.keys())
pprint.pprint(history.history)

# 学习曲线
# plt.plot(history.epoch, history.history.get("loss"), label="loss")
# plt.plot(history.epoch, history.history.get("val_loss"), label="val_loss")
# plt.plot(history.epoch, history.history.get("acc"), label="acc")
# plt.plot(history.epoch, history.history.get("val_acc"), label="val_acc")
# plt.legend()
# plt.show()
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# 预测
y_predict = model.predict(X_test[0].reshape(-1, 28, 28))
print(y_predict)
print(np.argmax(y_predict), y_test[0])

# 保存网络参数 下次使用前需要搭建好网络结构
model.save_weights("../save/model_weights.ckpt")
# 加载参数
model.load_weights("../save/model_weights.ckpt")
# 评估
model.evaluate(X_test, y_test)

# 保存网络参数+结构 下次使用前无需搭建好网络结构
model.save("../save/model.h5")
# 删除模型
del model
# 恢复模型
model_restore = tf.keras.models.load_model("../save/model.h5")
# 评估
model_restore.evaluate(X_test, y_test)
