import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

epoch = 10
batch_size = 32


def WriteCsv():
    housing = fetch_california_housing("../data/house")
    X = housing.data
    y = housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train)
    y_train_df = pd.DataFrame(y_train)
    train_csv = pd.concat([X_train_df, y_train_df], axis=1)
    X_test_df = pd.DataFrame(X_test)
    y_test_df = pd.DataFrame(y_test)
    test_csv = pd.concat([X_test_df, y_test_df], axis=1)

    column = housing.feature_names
    column.append("Label")

    train_csv.to_csv("../read/csv/train_csv.csv", float_format="%.2f",
                     index=False, header=column)  # 不保存index header=column
    test_csv.to_csv("../read/csv/test_csv.csv", float_format="%.2f",
                    index=False, header=column)  # 不保存index header=column


def ReadCsv(dir):
    # 解析函数
    def ParseLine(line):
        record = [np.nan] * 9  # 9个列 默认None
        parsed_fields = tf.io.decode_csv(line, record_defaults=record)  # 解析一行
        X = tf.stack(parsed_fields[0:-1])  # 分割出8个x
        y = tf.stack(parsed_fields[-1:])  # 分割出1个y
        return X, y

    # 得到csv文件
    dataset = tf.data.Dataset.list_files(dir)
    # 重复
    dataset = dataset.repeat(epoch)
    # 按行读取文本组成dataset 一对多
    dataset = dataset.interleave(lambda filename: tf.data.TextLineDataset(filename).skip(1),  # 跳过第一行column
                                 cycle_length=5, block_length=1)  # 并行处理5个 每次取1个
    # 乱序
    dataset = dataset.shuffle(10000)
    # 解析字符串转换为数字 多对多
    dataset = dataset.map(ParseLine)  # 把dataset用ParseLine函数操作后 再组成dataset
    # 批处理
    dataset = dataset.batch(batch_size)

    return dataset


if __name__ == '__main__':
    # 写入csv
    # WriteCsv()

    # 读取csv
    train_dataset = ReadCsv("../read/csv/train_csv.csv")
    test_dataset = ReadCsv("../read/csv/test_csv.csv")

    # 训练
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation="relu", input_shape=[8]))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.summary()

    model.compile(loss="mse", optimizer="adam")
    model.fit(train_dataset, epochs=epoch, steps_per_epoch=15480 // batch_size,
              validation_data=test_dataset, validation_steps=5160 // batch_size)

    model.evaluate(test_dataset, steps=5160 // batch_size)
