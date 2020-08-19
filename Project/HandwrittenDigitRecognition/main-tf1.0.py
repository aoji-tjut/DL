import numpy as np
import cv2 as cv
import tensorflow as tf
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
import sys

src = 0


# Window
class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()

        # 窗口初始化
        self.setWindowTitle("手写数字识别")
        self.setFixedSize(420, 120)

        # 文字1
        self.mylabel1 = QtWidgets.QLabel(self)
        self.mylabel1.setText("图片路径:")
        self.mylabel1.setFixedSize(65, 30)
        self.mylabel1.move(5, 15)

        # 文本框1
        self.mytext1 = QtWidgets.QTextEdit(self)
        self.mytext1.setReadOnly(True)
        self.mytext1.setFixedSize(290, 50)
        self.mytext1.move(65, 5)

        # Button1
        self.myButton1 = QtWidgets.QPushButton(self)
        self.myButton1.setFixedSize(65, 60)
        self.myButton1.move(355, 2)
        self.myButton1.setText("...")
        self.myButton1.clicked.connect(self.browse)

        # 文字2
        self.mylabel2 = QtWidgets.QLabel(self)
        self.mylabel2.setText("识别结果:")
        self.mylabel2.move(5, 70)
        self.mylabel2.setFixedSize(70, 30)

        # 文本框2
        self.mytext2 = QtWidgets.QLineEdit(self)
        self.mytext2.setReadOnly(True)
        self.mytext2.setFixedSize(290, 50)
        self.mytext2.move(65, 60)

        # Button2
        self.myButton2 = QtWidgets.QPushButton(self)
        self.myButton2.setFixedSize(65, 60)
        self.myButton2.move(355, 57)
        self.myButton2.setText("确定")
        self.myButton2.setEnabled(False)
        self.myButton2.clicked.connect(self.confirm)

    # 选择图片
    def browse(self):
        fileName, filetype = QFileDialog.getOpenFileName(self, "选择图片", "/")
        self.mytext1.setPlainText(fileName)

        if fileName == "":
            self.myButton2.setEnabled(False)
            self.mytext2.setText("")
            return

        global src
        src = cv.imread(fileName)

        try:
            src.data
        except Exception:
            self.mytext1.setText("此文件不是图片或文件读取失败！\n请重新选择！")
            self.myButton2.setEnabled(False)
            self.mytext2.setText("")
            return

        self.myButton2.setEnabled(True)

    # 处理+识别
    def confirm(self):
        result = tensorflow(opencv(src))
        self.mytext2.setText(str(int(result)))


# Tensorflow
def tensorflow(dst):
    def weight_variable(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias_variable(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def conv2d_same(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    def conv2d_valid(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    tf.reset_default_graph()  # 清除默认图的堆栈

    # learning_rate = tf.Variable(0.0001, dtype=tf.float32)  # 训练时学习率 测试中没用 但不能没有
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    x_image = tf.reshape(x, [-1, 28, 28, 1])  # [批次，高度，宽度，通道]

    # 第一个卷积层+池化层
    W_conv1 = weight_variable([5, 5, 1, 64])  # 5*5的采样窗口，输入通道1，输出通道64
    b_conv1 = bias_variable([64])  # 每个通道一个偏置值
    h_conv1 = tf.nn.relu(conv2d_valid(x_image, W_conv1) + b_conv1)  # [?,28,28,1]->[?,24,24,64]
    h_pool1 = max_pool_2x2(h_conv1)  # [?,24,24,64]->[?,12,12,64]

    # 第二个卷积层+池化层
    W_conv2 = weight_variable([3, 3, 64, 128])  # 3*3的采样窗口，输入通道64，输出通道128
    b_conv2 = bias_variable([128])  # 每个通道一个偏置值
    h_conv2 = tf.nn.relu(conv2d_same(h_pool1, W_conv2) + b_conv2)  # [?,12,12,64]->[?,12,12,128]
    h_pool2 = max_pool_2x2(h_conv2)  # [?,12,12,128]->[?,6,6,128]

    # 第一个全连接层
    h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 6 * 128])  # 扁平化 把四维[?,6,6,64]转化为二维[?,6*6*64]
    W_fc1 = weight_variable([6 * 6 * 128, 1024])  # 输入6*6*128个神经元 输出1024个神经元
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # [?,1024] = [?,6*6*128]*[6*6*128,1024] + [1024]
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 第二个全连接层
    W_fc2 = weight_variable([1024, 300])  # 输入1024个神经元 输出300个神经元
    b_fc2 = bias_variable([300])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # [?,300] = [?,1024]*[1024,300] + [300]
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # 第三个全连接层
    W_fc3 = weight_variable([300, 10])  # 十个分类
    b_fc3 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)  # [?,10] = [?,300]*[300,10] + [10]

    y_test = tf.argmax(prediction, 1)

    x_image = np.zeros([28, 28])
    y_label = np.zeros([1, 10], np.int)
    x_image[dst < 255] = 1
    x_image[dst == 255] = 0
    x_image = x_image.ravel().reshape(1, -1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./save/cnn/cnn.ckpt")
        result = sess.run(y_test, feed_dict={x: x_image, y: y_label, keep_prob: 1.0})

        return result


# OpenCV
def opencv(src):
    # 灰度
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)

    # 去除灰度值100以下的像素
    gray_new = np.zeros(gray.shape[:2], dtype=np.uint8) + 255
    gray_new[gray < 100] = 0
    # cv.imshow("gray_new", gray_new)

    # 去噪
    blur = cv.medianBlur(gray_new, 5)
    # cv.imshow("blur", blur)

    # 加粗
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    erode = cv.erode(blur, kernel)
    # cv.imshow("erode",erode)

    # 二值化
    ret, bin = cv.threshold(erode, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
    bin = bin if (np.sum(bin == 0) < np.sum(bin == 255)) else ~bin
    # cv.imshow("bin", bin)

    # 轮廓
    img, contours, hierarchy = cv.findContours(bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 寻找最大轮廓
    flag = -1
    last_area = 0
    for i, contour in enumerate(contours):
        # 不考虑图片轮廓
        if i == 0:
            continue

        area = cv.contourArea(contour)

        if area > last_area:
            flag = i
            last_area = area

    # 构造28*28数字
    global dst
    dst = np.zeros((28, 28), np.uint8) + 255
    for i, contour in enumerate(contours):
        if flag == -1:
            exit(0)

        if i == flag:
            # 分割出数字
            rect_x, rect_y, rect_w, rect_h = cv.boundingRect(contour)
            number = bin[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]
            # cv.imshow("number", number)

            # 把数字居中放到16*16的图中
            size = rect_h if (rect_h > rect_w) else rect_w
            x_pos = round(size / 2) - round(rect_h / 2)
            y_pos = round(size / 2) - round(rect_w / 2)
            temp = np.zeros((size, size), np.uint8) + 255
            temp[x_pos:x_pos + rect_h, y_pos:y_pos + rect_w] = number
            temp = cv.resize(temp, (50, 50))
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            temp = cv.erode(temp, kernel)
            temp = cv.resize(temp, (16, 16))
            # cv.imshow("temp", temp)

            # 把数字居中放到28*28的图中
            dst[6:22, 6:22] = temp
            cv.imwrite("image/test.jpeg", dst)
            # cv.imshow("dst", dst)

            return dst

    # cv.waitKey(0)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
