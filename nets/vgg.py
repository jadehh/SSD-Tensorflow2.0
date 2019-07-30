#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/7/29 by jade
# 邮箱：jadehh@live.com
# 描述：tensorflow 2.0 keras 的使用
# 最近修改：2019/7/29  下午5:36 modify by jade

import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Model

class VGG(Model):
    def __init__(self):
        super(VGG,self).__init__()
        self.conv1_1 = Conv2D(64,3,padding="SAME",activation='relu',name="conv1_1")
        self.pool1_1 = MaxPool2D(2,name="pool1_1")
        self.conv1_2 = Conv2D(64,3,padding="SAME",activation='relu',name="conv1_2")
        self.pool1_2 = MaxPool2D(2, name="pool1_2")

        self.conv2_1 = Conv2D(128,3,padding="SAME",activation='relu',name="conv2_1")
        self.pool2_1 = MaxPool2D(2,name="pool2_1")
        self.conv2_2 = Conv2D(128,3,padding="SAME",activation='relu',name="conv2_2")
        self.pool2_2 = MaxPool2D(2,name="pool2_2")


    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.pool1_1(x)
        x = self.conv1_2(x)
        x = self.poo1_2(x)

        x = self.conv2_1(x)
        x = self.pool2_1(x)
        x = self.conv2_2(x)
        x = self.pool2_2(x)


        print x


class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(64, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


if __name__ == '__main__':
    model = VGG()
    images = tf.constant(1,shape=[16,224,224,3],dtype=tf.float32)
    predictions = model(images)