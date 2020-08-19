import tensorflow as tf
import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt

epoch = 30
batch_size = 32


def LoadData():
    train = pd.read_csv("../data/titanic/train.csv")
    test = pd.read_csv("../data/titanic/test.csv")

    # 删除任何有缺失值的数据
    if np.any(train.isnull() == True):
        train.dropna(axis=0, how="any")
    if np.any(test.isnull() == True):
        test.dropna(axis=0, how="any")

    print("train.shape =", train.shape)
    print("test.shape =", test.shape)

    # 取出survived列作为label 剩下的作为data
    y_train = train.pop("survived")
    y_test = test.pop("survived")
    X_train = train
    X_test = test

    print("X_train.shape =", X_train.shape, "y_train.shape =", y_train.shape)
    print("X_test.shape =", X_test.shape, "y_test.shape =", y_test.shape)
    print("X_train.head(5) =\n", X_train.head(5))
    print("y_train.head(5) =\n", y_train.head(5))

    return X_train, y_train, X_test, y_test


def ShowData(X_train, y_train):
    # age直方图
    plt.figure("age")
    X_train["age"].hist(bins=20)  # 分成20个区间

    # sex柱状图
    plt.figure("sex")
    X_train["sex"].value_counts().plot(kind="barh")  # 横向画图

    # class柱状图
    plt.figure("class")
    X_train["class"].value_counts().plot(kind="bar")  # 纵向画图

    # 按性别分组 计算幸存者的均值(比例)
    proportion = pd.concat([X_train, y_train], axis=1).groupby("sex")["survived"].mean()
    print(proportion)
    plt.figure("proportion")
    proportion.plot(kind="barh")

    plt.show()


def MakeDataset(X, y):
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)

    return dataset


def FeatureProcessing(X_train):
    # 离散型特征(有固定分类的) 需要One-Hot编码后输入模型
    categorial_columns = ["sex", "n_siblings_spouses", "parch", "class", "deck", "embark_town", "alone"]
    # 连续型特征 直接输入模型
    numeric_columns = ["age", "fare"]
    # 特征集合
    feature_columns = []

    # 处理离散特征
    for feature in categorial_columns:
        # 获得feature出现过的值 [*,*,*,...]
        vocabulary = X_train[feature].unique()
        # 结合为VocabularyListCategoricalColumn
        list = tf.feature_column.categorical_column_with_vocabulary_list(feature, vocabulary)
        # 获得One-Hot编码 IndicatorColumn
        one_hot = tf.feature_column.indicator_column(list)
        # 存入feature_columns列表
        feature_columns.append(one_hot)

    # 处理连续特征
    for feature in numeric_columns:
        # 获得NumericColumn shape=[1,]
        list = tf.feature_column.numeric_column(feature, dtype=tf.float32)
        # 存入feature_columns列表
        feature_columns.append(list)

    return feature_columns


if __name__ == '__main__':
    # 加载数据
    X_train, y_train, X_test, y_test = LoadData()

    # 可视化
    # ShowData(X_train, y_train)

    # 构建dataset
    dataset_train = MakeDataset(X_train, y_train)
    dataset_test = MakeDataset(X_test, y_test)

    # 获得feature_columns
    feature_columns = FeatureProcessing(X_train)
    pprint.pprint(feature_columns)
    print("len(feature_columns) =", len(feature_columns))

    # 训练
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.DenseFeatures(feature_columns, name="feature_columns"))  # [-1,34] 2+7+6+3+8+4+2+1+1=34
    model.add(tf.keras.layers.Dense(128, activation="relu", name="dense1"))
    model.add(tf.keras.layers.Dense(64, activation="relu", name="dense2"))
    model.add(tf.keras.layers.Dense(32, activation="relu", name="dense3"))
    model.add(tf.keras.layers.Dense(2, activation="sigmoid", name="output"))  # 输出生/死概率

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])
    callback = [tf.keras.callbacks.TensorBoard("../board")]
    model.fit(dataset_train, epochs=epoch, steps_per_epoch=627 // batch_size, callbacks=callback,
              validation_data=dataset_test, validation_steps=264 // batch_size)

    model.summary()  # DenseFeatures只有在fit后才能显示

    # 分析predict
    X, y = 0, 0
    for i, j in dataset_test.take(1):  # 取一个batch_size个数据
        X = i
        y = j
    probability = model.predict(X)
    print("probability =\n", probability)
    print("probability_shape =", probability.shape)
    print("y_shape =", y.numpy().shape)
    print("predict =", probability.argmax(axis=1))
    print("      y =", y.numpy())
    print(np.sum(probability.argmax(axis=1) == y.numpy()) / batch_size)  # 一个batch_size的准确率
