#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/7/29 by jade
# 邮箱：jadehh@live.com
# 描述：tensorflow 2.0 keras 的使用
# 最近修改：2019/7/29  下午5:36 modify by jade

import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Model
from jade import *

from tensorflow.python import keras

from datasetopeation.jadeClassifyTFRecords import LoadClassifyTFRecord,CreateClassifyTFRecorder

class VGGBase(Model):
    def __init__(self):
        super(VGGBase, self).__init__()
        # conv1 两次卷积 + MaxPool
        self.conv1_1 = Conv2D(64, (3, 3), padding="same", activation='relu', name="conv1_1")
        self.conv1_2 = Conv2D(64, (3, 3), padding="same", activation='relu', name="conv1_2")
        self.pool1 = MaxPool2D((2, 2), strides=(2, 2), padding="same",name="pool1")
        # conv2 两次卷积 + MaxPool
        self.conv2_1 = Conv2D(128, (3, 3), padding="same", activation='relu', name="conv2_1")
        self.conv2_2 = Conv2D(128, (3, 3), padding="same", activation='relu', name="conv2_2")
        self.pool2 = MaxPool2D((2, 2), strides=(2, 2), padding="same",name="pool2")
        # conv3 三次卷积 + MaxPool
        self.conv3_1 = Conv2D(256, (3, 3), padding="same", activation='relu', name="conv3_1")
        self.conv3_2 = Conv2D(256, (3, 3), padding="same", activation='relu', name="conv3_2")
        self.conv3_3 = Conv2D(256, (3, 3), padding="same", activation='relu', name="conv3_3")
        self.pool3 = MaxPool2D((2, 2), strides=(2, 2), padding="same",name="pool3")

        # conv4 三次卷积 + MaxPool
        self.conv4_1 = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv4_1")
        self.conv4_2 = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv4_2")
        self.conv4_3 = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv4_3")
        self.pool4 = MaxPool2D((2, 2), strides=(2, 2), padding="same",name="pool4")
        # conv5 三次卷积 + MaxPool
        self.conv5_1 = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv5_1")
        self.conv5_2 = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv5_2")
        self.conv5_3 = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv5_3")

    def call(self,x):
        # conv1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        # conv2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        # conv3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        # conv4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        endpoints = x
        x = self.pool4(x)
        # conv5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        return x,endpoints


class VGGNet16(Model):
    def __init__(self, classes=10):
        super(VGGNet16, self).__init__()
        self.vgg_base = VGGBase()
        self.pool5 = MaxPool2D((2, 2), strides=(2, 2),padding="same", name="pool5")
        self.flatten = Flatten(name='flatten')
        self.fc6 = Dense(4096, activation='relu', name='fc6')
        # fc7
        self.fc7 = Dense(4096, activation='relu', name='fc7')
        # fc8
        self.fc8 = Dense(classes, activation='softmax', name='fc8')

    # 向前传播foward
    def call(self, x):

        x,_ = self.vgg_base(x)
        x = self.pool5(x)
        x = self.flatten(x)
        # fc6
        x = self.fc6(x)
        # fc7
        x = self.fc7(x)
        # fc8
        x = self.fc8(x)

        return x

#全卷积层
class VGGNetConv(Model):
    def __init__(self, classes=10):
        super(VGGNetConv, self).__init__()
        # conv1 两次卷积 + MaxPool
        self.conv1_1 = Conv2D(64, (3, 3), padding="same", activation='relu', name="conv1_1")
        self.conv1_2 = Conv2D(64, (3, 3), padding="same", activation='relu', name="conv1_2")
        self.pool1 = MaxPool2D((2, 2), strides=(2, 2), name="pool1")
        # conv2 两次卷积 + MaxPool
        self.conv2_1 = Conv2D(128, (3, 3), padding="same", activation='relu', name="conv2_1")
        self.conv2_2 = Conv2D(128, (3, 3), padding="same", activation='relu', name="conv2_2")
        self.pool2 = MaxPool2D((2, 2), strides=(2, 2), name="pool2")
        # conv3 三次卷积 + MaxPool
        self.conv3_1 = Conv2D(256, (3, 3), padding="same", activation='relu', name="conv3_1")
        self.conv3_2 = Conv2D(256, (3, 3), padding="same", activation='relu', name="conv3_2")
        self.conv3_3 = Conv2D(256, (3, 3), padding="same", activation='relu', name="conv3_3")
        self.pool3 = MaxPool2D((2, 2), strides=(2, 2), name="pool3")

        # conv4 三次卷积 + MaxPool
        self.conv4_1 = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv4_1")
        self.conv4_2 = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv4_2")
        self.conv4_3 = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv4_3")
        self.pool4 = MaxPool2D((2, 2), strides=(2, 2), name="pool4")
        # conv5 三次卷积 + MaxPool
        self.conv5_1 = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv5_1")
        self.conv5_2 = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv5_2")
        self.conv5_3 = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv5_3")
        self.pool5 = MaxPool2D((2, 2), strides=(2, 2), name="pool5")

        self.fc6 = Conv2D(4096,(7,7),padding="valid",activation='relu',name='conv6')
        self.dropout6 = Dropout(0.5)

        self.fc7 = Conv2D(4096,(1,1),activation='relu',name='conv7')
        self.dropout7 = Dropout(0.5)

        self.fc8 = Conv2D(classes,(1,1),padding='same',activation=None,name='conv8')


    # 向前传播foward
    def call(self, x):
        # conv1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        # conv2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        # conv3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        # conv4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        # conv5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        # fc6
        x = self.fc6(x)
        x = self.dropout6(x)
        # fc7
        x = self.fc7(x)
        x = self.dropout7(x)
        # fc8
        x = self.fc8(x)

        x = tf.squeeze(x, [1,2],name='fc8/squeezed')
        return x