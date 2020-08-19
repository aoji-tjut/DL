import tensorflow as tf
import cv2 as cv
import numpy as np
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
        self.setFixedSize(450, 105)

        # 文字1
        self.mylabel1 = QtWidgets.QLabel(self)
        self.mylabel1.setText("图片路径:")
        self.mylabel1.setFixedSize(65, 25)
        self.mylabel1.move(5, 15)

        # 文本框1
        self.mytext1 = QtWidgets.QTextEdit(self)
        self.mytext1.setReadOnly(True)
        self.mytext1.setFixedSize(325, 45)
        self.mytext1.move(65, 5)

        # Button1
        self.myButton1 = QtWidgets.QPushButton(self)
        self.myButton1.setFixedSize(60, 53)
        self.myButton1.move(390, 2)
        self.myButton1.setText("...")
        self.myButton1.clicked.connect(self.browse)

        # 文字2
        self.mylabel2 = QtWidgets.QLabel(self)
        self.mylabel2.setText("识别结果:")
        self.mylabel2.setFixedSize(70, 30)
        self.mylabel2.move(5, 60)

        # 文本框2
        self.mytext2 = QtWidgets.QLineEdit(self)
        self.mytext2.setReadOnly(True)
        self.mytext2.setFixedSize(15, 45)
        self.mytext2.move(65, 55)

        # 文字3
        self.mylabel3 = QtWidgets.QLabel(self)
        self.mylabel3.setText("识别概率:")
        self.mylabel3.setFixedSize(70, 30)
        self.mylabel3.move(85, 60)

        # 文本框3
        self.mytext3 = QtWidgets.QTextEdit(self)
        self.mytext3.setReadOnly(True)
        self.mytext3.setFixedSize(245, 45)
        self.mytext3.move(145, 55)

        # Button2
        self.myButton2 = QtWidgets.QPushButton(self)
        self.myButton2.setFixedSize(60, 53)
        self.myButton2.move(390, 52)
        self.myButton2.setText("确定")
        self.myButton2.setEnabled(False)
        self.myButton2.clicked.connect(self.confirm)

    # 选择图片
    def browse(self):
        fileName, filetype = QFileDialog.getOpenFileName(self, "选择图片", "./")
        self.mytext1.setPlainText(fileName)

        if fileName == "":
            self.myButton2.setEnabled(False)
            self.mytext2.setText("")
            self.mytext3.setText("")
            return

        global src
        src = cv.imread(fileName)

        try:
            src.data
        except Exception:
            self.mytext1.setText("此文件不是图片或文件读取失败！\n请重新选择！")
            self.myButton2.setEnabled(False)
            self.mytext2.setText("")
            self.mytext3.setText("")
            return

        self.myButton2.setEnabled(True)

    # 处理+识别
    def confirm(self):
        predict, probability = tensorflow(opencv(src))
        self.mytext2.setText(predict)
        self.mytext3.setText(probability)


# Tensorflow
def tensorflow(dst):
    dst = ~dst
    dst = dst.reshape(1, 28, 28, 1)

    model = tf.keras.models.load_model("./save/cnn.h5")
    predict = model.predict(dst)
    predict = predict.reshape(-1)

    # 概率字符串
    probability = "/"
    for num, value in enumerate(predict):
        probability = probability + "%d:" % num + "%.2f/" % value
        if num == 4:
            probability = probability + "\n/"

    # 预测结果
    predict = np.argmax(predict)

    return str(predict), probability


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
