#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/7/29 by jade
# 邮箱：jadehh@live.com
# 描述：tensorflow 2.0 keras 的使用
# 最近修改：2019/7/29  下午5:36 modify by jade

import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Model, optimizers
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.datasets import cifar100
from tensorflow.python.keras.applications.vgg16 import VGG16
import cv2

from tensorflow.python import keras
import numpy as np
from jade import *
from jade.jadeTFRecords import *

class VGGNetDense(Model):
    def __init__(self, classes=10):
        super(VGGNetDense, self).__init__()
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

        self.flatten = Flatten(name='flatten')

        self.fc6 = Dense(4096, activation='relu', name='fc6')
        # fc7
        self.fc7 = Dense(4096, activation='relu', name='fc7')
        # fc8
        self.fc8 = Dense(classes, activation='softmax', name='fc8')

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

        x = self.flatten(x)
        # fc6
        x = self.fc6(x)
        # fc7
        x = self.fc7(x)
        # fc8
        x = self.fc8(x)

        return x


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


def LoadTFRecord(tfrecord_path, batch_size=32, shuffle=True,repeat=True):
    return loadClassifyTFRecord(tfrecord_path, batch_size, shuffle,repeat)

def createTFRecord():
    CreateClassTFRecorder("/home/jade/Data/sdfgoods10_224","sdfgoods10_224")
def train():
    train_batch_generator = LoadTFRecord("/home/jade/Data/TFRecords/sdfgoods10_224_train.tfrecord")
    test_batch_generator = LoadTFRecord("/home/jade/Data/TFRecords/sdfgoods10_224_test.tfrecord")
    # x_train,y_train,x_test,y_test = loadDataSet()
    model = VGGNetConv(classes=10)
    model.load_weights("VGGNetConv")
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  # loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit_generator(generator=train_batch_generator,
                                  steps_per_epoch=100,
                                  epochs=10,
                                  verbose=1,
                                  validation_data=test_batch_generator,
                                  validation_steps=1)
    model.summary()
    model.save_weights('VGGNetDense', save_format='tf')


def predict():
    # 读取保存的模型参数
    new_model = VGGNet(classes=10)
    # new_model.train_on_batch(x_train[:1], y_train[:1])
    new_model.load_weights('VGG')
    test_generator = loadClassifyTFRecord("/home/jade/Data/TFRecords/sdfgoods10_test.tfrecord",repeat=False)
    num = 0
    correct = 0
    while (True):
        try:
            x_test, y_test = next(test_generator)
            new_predictions = new_model.predict(x_test)
            for i in range(new_predictions.shape[0]):
                num = num + 1
                print(np.argmax(new_predictions[i]),y_test[i].numpy())
                if np.argmax(new_predictions[i]) == y_test[i].numpy():
                    correct = correct + 1

        except:
            break
    print(correct/float(num))








if __name__ == '__main__':
    # createTFRecord()
    train()
    # predict()
