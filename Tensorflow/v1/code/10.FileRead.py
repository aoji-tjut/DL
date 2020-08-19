import os
import tensorflow as tf


def ReadCsv(file_dir):
    # 文件队列
    file_queue = tf.train.string_input_producer(file_dir)

    # 阅读器
    reader = tf.TextLineReader()  # 一次读取一行
    key, value = reader.read(file_queue)

    # 解码
    x, y = tf.decode_csv(value, record_defaults=[[0], ["None"]])  # 第一列int类型默认0 第二列string类型默认"None"

    # 批处理
    x_batch, y_batch = tf.train.batch([x, y], batch_size=20, num_threads=1, capacity=20)  # tensor 批次大小 线程数 队列大小

    return x_batch, y_batch


def ReadImage(file_dir):
    # 文件队列
    file_queue = tf.train.string_input_producer(file_dir)

    # 阅读器
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)

    # 解码
    image = tf.image.decode_png(value, channels=3)  # 一次读取一张

    # 统一图片大小
    image = tf.image.resize(image, [100, 100])

    # 批处理
    image_batch = tf.train.batch([image], batch_size=20, num_threads=1, capacity=20)

    return image_batch


def ReadBin(file_dir):
    '''
    每个样本格式：[y,x]
    x: [32,32,3]
    y: [1]
    bytes: 32*32*3+1=3073
    '''

    # 文件队列
    file_queue = tf.train.string_input_producer(file_dir)

    # 阅读器
    reader = tf.FixedLengthRecordReader(3073)  # 一次读取一个样本 3073字节(x+y)
    key, value = reader.read(file_queue)

    # 解码
    image = tf.decode_raw(value, tf.uint8)  # [0]~[3072]

    # 分割label/x
    y = tf.slice(image, [0], [1])  # [0]
    x = tf.slice(image, [1], [3072])  # [1]~[3072]

    # 改变data形状[3072]->[32,32,3]
    x = tf.reshape(x, [32, 32, 3])

    # 批处理
    x_batch, y_batch = tf.train.batch([x, y], batch_size=20, num_threads=1, capacity=20)

    return x_batch, y_batch


def WriteTFRecord(bin_x_batch, bin_y_batch):
    print("TFReacord开始写入")

    # 建立TFRecord存储器
    writer = tf.python_io.TFRecordWriter("../read/tfrecord/tfrecord.tfrecords")

    # 将所有样本写入
    for i in range(tf.shape(bin_x_batch).eval()[0]):  # 样本个数
        # 得到样本数据
        image = bin_x_batch[i].eval().tostring()  # string
        label = bin_y_batch[i].eval()[0]  # int

        # 构造example
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))

        # 写入单独样本
        writer.write(example.SerializeToString())

    # 关闭
    writer.close()
    print("TFReacord写入完毕")


def ReadTFRecord():
    # 文件队列
    file_queue = tf.train.string_input_producer(["../read/tfrecord/tfrecord.tfrecords"])

    # 阅读器
    reader = tf.TFRecordReader()  # 一次读取一个样本
    key, value = reader.read(file_queue)

    # 解析example
    features = tf.parse_single_example(value, features={
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    })

    # 获取数据
    image = tf.decode_raw(features["image"], tf.uint8)  # 只有string需要解码
    label = features["label"]

    # 统一形状
    image = tf.reshape(image, [32, 32, 3])

    # 批处理
    image_batch, labels_batch = tf.train.batch([image, label], batch_size=20, num_threads=1, capacity=20)

    return image_batch, labels_batch


if __name__ == '__main__':
    # 得到目录下文件
    file_csv = os.listdir("../read/csv")
    file_image = os.listdir("../read/image")
    file_bin = os.listdir("../read/bin")
    # 得到文件目录
    csv_dir = [os.path.join("../read/csv", file) for file in file_csv if file[-3:] == "csv"]
    image_dir = [os.path.join("../read/image", file) for file in file_image if file[-3:] == "png"]
    bin_dir = [os.path.join("../read/bin", file) for file in file_bin if file[-3:] == "bin"]
    print(csv_dir)
    print(image_dir)
    print(bin_dir)

    # 得到数据
    csv_x_batch, csv_y_batch = ReadCsv(csv_dir)
    image_batch = ReadImage(image_dir)
    bin_x_batch, bin_y_batch = ReadBin(bin_dir)
    tfrecord_image_batch, tfrecord_label_batch = ReadTFRecord()

    with tf.Session() as sess:
        # 线程协调器
        coord = tf.train.Coordinator()
        # 开启读取文件线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # ..........................................................
        print(sess.run([csv_x_batch, csv_y_batch]))
        print("......................................................................................................")
        print(sess.run(image_batch))
        print("......................................................................................................")
        print(sess.run([bin_x_batch, bin_y_batch]))
        print("......................................................................................................")
        # WriteTFRecord(bin_x_batch, bin_y_batch)
        print("......................................................................................................")
        print(sess.run([tfrecord_image_batch, tfrecord_label_batch]))
        # ..........................................................

        # 回收线程
        coord.request_stop()
        coord.join(threads)
