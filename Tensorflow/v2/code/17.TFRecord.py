import tensorflow as tf
import numpy as np

epoch = 10
batch_size = 32


# 读取csv
def ReadCsv(dir):
    # 解码函数
    def ParseLine(line):
        record = [np.nan] * 9  # 9个列 默认None
        parsed_fields = tf.io.decode_csv(line, record_defaults=record)
        X = tf.stack(parsed_fields[0:-1])  # 分割出8个x
        y = tf.stack(parsed_fields[-1:])  # 分割出1个y
        return X, y

    # 得到csv文件
    dataset = tf.data.Dataset.list_files(dir)
    # 读取文件为字符串
    dataset = dataset.interleave(lambda filename: tf.data.TextLineDataset(filename).skip(1),  # 跳过第一行
                                 cycle_length=5, block_length=1)  # 并行处理5个 每次取1个
    # 解码 字符串转换为数字
    dataset = dataset.map(ParseLine)  # 把dataset用ParseLine函数操作后 再组成dataset
    # 批处理
    dataset = dataset.batch(batch_size)

    return dataset


# 写入TFRecord
def WriteTFRecord(dir, dataset, steps_per_shard, compression_type=None):
    # 序列化函数
    def SerializeExample(X, y):
        X_list = tf.train.FloatList(value=X)
        y_list = tf.train.FloatList(value=y)
        features = tf.train.Features(
            feature={
                'X': tf.train.Feature(float_list=X_list),
                'y': tf.train.Feature(float_list=y_list)
            }
        )
        example = tf.train.Example(features=features)
        examply_serialize = example.SerializeToString()
        return examply_serialize

    options = tf.io.TFRecordOptions(compression_type=compression_type)
    with tf.io.TFRecordWriter(dir, options=options) as writer:
        # 读取csv时每批32个 一共有steps_per_shard批
        for X_batch, y_batch in dataset.take(steps_per_shard):
            # 处理一个批次 共32个数据
            for X_example, y_example in zip(X_batch, y_batch):
                # 序列化每一个数据 变成乱码字符串
                examply_serialize = SerializeExample(X_example, y_example)
                writer.write(examply_serialize)


# 读取TFRecord
def ReadTFRecord(dir, compression_type=None):
    features = {
        'X': tf.io.FixedLenFeature([8], dtype=tf.float32),
        'y': tf.io.FixedLenFeature([1], dtype=tf.float32)
    }

    # 解码函数
    def ParseExample(example_serialize):
        # 把序列化的example_serialize解码为exmaple字典
        example = tf.io.parse_single_example(example_serialize, features)
        # 取出字典中的值
        X = example['X']
        y = example['y']
        return X, y

    # 得到csv文件
    dataset = tf.data.Dataset.list_files(dir)
    # 重复
    dataset = dataset.repeat(epoch)
    # 读取文件为字符串
    dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(filename, compression_type=compression_type),
                                 cycle_length=5, block_length=1)  # 并行处理5个 每次取1个
    # 乱序
    dataset = dataset.shuffle(10000)
    # 解码 字符串转换为数字
    dataset = dataset.map(ParseExample)  # 把dataset用ParseExample函数操作后 再组成dataset
    # 批处理
    dataset = dataset.batch(batch_size)

    return dataset


if __name__ == '__main__':
    # 从csv中读取数据 注意不要repeat()和shuffle()
    # train_dataset = ReadCsv("../read/csv/train_csv.csv")
    # test_dataset = ReadCsv("../read/csv/test_csv.csv")

    # 将读取出的数据写入tfrecord
    # WriteTFRecord("../read/tfrecord/train.tfrecords.zip", train_dataset, 15480 // batch_size, compression_type="GZIP")
    # WriteTFRecord("../read/tfrecord/test.tfrecords.zip", test_dataset, 5160 // batch_size, compression_type="GZIP")

    # 从tfrecord中读取数据
    train_dataset = ReadTFRecord("../read/tfrecord/train.tfrecords.zip", compression_type="GZIP")
    test_dataset = ReadTFRecord("../read/tfrecord/test.tfrecords.zip", compression_type="GZIP")

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
