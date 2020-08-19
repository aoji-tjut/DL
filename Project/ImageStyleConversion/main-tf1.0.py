import os
import numpy as np
import cv2 as cv
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]  # b、g、r通道均值
vgg_path = "./vgg/vgg16.npy"
content_img_path = "./image/Night.jpg"
style_img_path = "./image/StarSky.jpg"
output_path = "./image/result"
num_steps = 100
learning_rate = 10
content_coef = 0.01  # 图片内容损失系数
style_coef = 100  # 图片风格损失系数


class VGGNet:
    def __init__(self, data_dict):
        self.data_dict = data_dict

    # 得到卷积层过滤器
    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="conv")

    # 得到全连接层权值
    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="fc")

    # 得到偏置值
    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="bias")

    # 卷积层
    def conv_layer(self, input, name):
        with tf.name_scope(name):
            conv_w = self.get_conv_filter(name)
            conv_b = self.get_bias(name)
            z = tf.nn.conv2d(input, filter=conv_w, strides=[1, 1, 1, 1], padding="SAME") + conv_b
            a = tf.nn.relu(z)
            return a

    # 池化层
    def pool_layer(self, input, name):
        with tf.name_scope(name):
            return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    # 展平层
    def flat_layer(self, input, name):
        with tf.name_scope(name):
            shape = input.get_shape()
            input_reshape = tf.reshape(input, [-1, shape[1] * shape[2] * shape[3]])
            return input_reshape

    # 全连接层
    def fc_layer(self, input, name, activation=tf.nn.relu):
        with tf.name_scope(name):
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            z = tf.matmul(input, fc_w) + fc_b
            a = activation(z)
            return a

    # 构建网络
    def build(self, x_rbg):
        # 图像预处理
        r, g, b = tf.split(x_rbg, [1, 1, 1], axis=3)
        x_bgr = tf.concat([b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)
        assert x_bgr.get_shape()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(x_bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.pool_layer(self.conv1_2, "pool1")

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.pool_layer(self.conv2_2, "pool2")

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.pool_layer(self.conv3_3, "pool3")

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.pool_layer(self.conv4_3, "pool4")

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.pool_layer(self.conv5_3, "pool5")

        # self.flat = self.flat_layer(self.pool5, "falt")
        # self.fc6 = self.fc_layer(self.flat, "fc6")
        # self.fc7 = self.fc_layer(self.fc6, "fc7")
        # self.fc8 = self.fc_layer(self.fc7, "fc8", tf.nn.softmax)
        return None


# 初始化结果图像
def initial_result(shape, mean, stddev):
    initial = tf.truncated_normal(shape, mean, stddev)
    return tf.Variable(initial)


# 读取图像
def read_image(path):
    img = cv.imread(path)
    img = cv.resize(img, (224, 224))
    img = np.array(img)
    img = img.reshape([1, 224, 224, 3])
    return img


# 计算gram矩阵
def gram_matrix(x):
    batch, height, width, channel = x.get_shape().as_list()  # 转换成列表
    features = tf.reshape(x, [batch, height * width, channel])
    # 计算矩阵相似度 [channel,height*width] * [height*width,channel] = [channel,channel]
    gram = tf.matmul(features, features, adjoint_a=True)  # 对第一个矩阵进行转置
    gram = gram / tf.constant(height * width * channel, dtype=tf.float32)  # 避免值过大
    return gram


if __name__ == '__main__':
    content_img = read_image(content_img_path)  # 内容图像
    style_img = read_image(style_img_path)  # 风格图像
    result_img = initial_result([1, 224, 224, 3], 128, 20)  # 输出图像
    content_x = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
    style_x = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])

    # 加载vgg模型
    vgg = np.load(vgg_path, allow_pickle=True, encoding="latin1")
    data_dict = vgg.item()  # 返回字典
    # import pprint
    # pprint.pprint(data_dict)

    # 创建对象
    vgg_content = VGGNet(data_dict)
    vgg_style = VGGNet(data_dict)
    vgg_result = VGGNet(data_dict)

    # 构造网络
    vgg_content.build(content_x)
    vgg_style.build(style_x)
    vgg_result.build(result_img)

    # 提取特征 可以增加层
    content_features = [vgg_content.conv1_2]  # 内容特征越低层越好
    result_content_features = [vgg_result.conv1_2]  # 结果图像内容特征
    style_features = [vgg_style.conv4_3]  # 风格特征越高层越好
    result_style_features = [vgg_result.conv4_3]  # 结果图像风格特征

    # 计算content损失
    content_loss = tf.zeros(1, tf.float32)
    for cf, rcf in zip(content_features, result_content_features):
        content_loss += tf.reduce_mean(tf.square(cf - rcf), axis=[1, 2, 3])  # 在[height,width,channel]上计算损失

    # 计算风格特征gram矩阵
    style_gram = [gram_matrix(feature) for feature in style_features]
    result_style_gram = [gram_matrix(feature) for feature in result_style_features]

    # 计算style损失
    style_loss = tf.zeros(1, tf.float32)
    for sg, rsg in zip(style_gram, result_style_gram):
        style_loss += tf.reduce_mean(tf.square(sg - rsg), axis=[1, 2])  # 在[height*width,channel]上计算损失

    # 加权损失
    loss = content_coef * content_loss + style_coef * style_loss

    # 训练
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # writer = tf.summary.FileWriter("../board/", sess.graph)

        for step in range(num_steps):
            c_l, s_l, l, _ = sess.run([content_loss, style_loss, loss, train],
                                      feed_dict={content_x: content_img, style_x: style_img})
            print("step = %d, content_loss = %f, style_loss = %f, loss = %f" % (step + 1, c_l[0], s_l[0], l[0]))

            # 存储图片
            result_img_path = os.path.join(output_path, "result-%d.jpg" % (step + 1))  # 输出图片路径
            dst = result_img.eval()[0]  # 取出数据[1,224,224,3]->[224,224,3]
            dst = np.clip(dst, 0, 255)  # 值裁剪
            dst = np.array(dst, dtype=np.int)  # 类型转换
            cv.imwrite(result_img_path, dst)
