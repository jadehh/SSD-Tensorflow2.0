#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/7/29 by jade
# 邮箱：jadehh@live.com
# 描述：tensorflow 2.0 keras 的使用
# 最近修改：2019/7/29  下午5:36 modify by jade

import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Model,optimizers
from tensorflow.python.keras.datasets import cifar100
from tensorflow.python.keras.applications.vgg16 import VGG16
import cv2
from tensorflow.python import keras
import numpy as np
from jade import *
class MyModel(Model):
  def __init__(self,num_classes=100):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(num_classes, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

class VGGNet(Model):
    def __init__(self,classes=10):
        super(VGGNet,self).__init__()
        #conv1 两次卷积 + MaxPool
        self.conv1_1 = Conv2D(64,(3,3),padding="same",activation='relu',name="conv1_1")
        self.conv1_2 = Conv2D(64,(3,3),padding="same",activation='relu',name="conv1_2")
        self.pool1 = MaxPool2D((2, 2), strides=(2, 2),name="pool1")
        #conv2 两次卷积 + MaxPool
        self.conv2_1 = Conv2D(128,(3,3),padding="same",activation='relu',name="conv2_1")
        self.conv2_2 = Conv2D(128,(3,3),padding="same",activation='relu',name="conv2_2")
        self.pool2 = MaxPool2D((2, 2), strides=(2, 2),name="pool2")
        #conv3 三次卷积 + MaxPool
        self.conv3_1 = Conv2D(256,(3,3),padding="same",activation='relu',name="conv3_1")
        self.conv3_2 = Conv2D(256,(3,3),padding="same",activation='relu',name="conv3_2")
        self.conv3_3 = Conv2D(256,(3,3),padding="same",activation='relu',name="conv3_3")
        self.pool3 = MaxPool2D((2, 2), strides=(2, 2),name="pool3")

        #conv4 三次卷积 + MaxPool
        self.conv4_1 = Conv2D(512,(3,3),padding="same",activation="relu",name="conv4_1")
        self.conv4_2 = Conv2D(512,(3,3),padding="same",activation="relu",name="conv4_2")
        self.conv4_3 = Conv2D(512,(3,3),padding="same",activation="relu",name="conv4_3")
        self.pool4 = MaxPool2D((2, 2), strides=(2, 2), name="pool4")
        #conv5 三次卷积 + MaxPool
        self.conv5_1 = Conv2D(512,(3,3),padding="same",activation="relu",name="conv5_1")
        self.conv5_2 = Conv2D(512,(3,3),padding="same",activation="relu",name="conv5_2")
        self.conv5_3 = Conv2D(512,(3,3),padding="same",activation="relu",name="conv5_3")
        self.pool5 = MaxPool2D((2, 2), strides=(2, 2), name="pool5")

        #SSD这里使用卷积层代替了全连接层
        #使用卷积替代全连接层
        self.flatten = Flatten(name='flatten')
        self.fc6 = Dense(4096, activation='relu', name='fc6')
        #fc7
        self.fc7 = Dense(4096, activation='relu', name='fc7')
        #fc8
        self.fc8 =Dense(classes, activation='softmax', name='fc8')







    #向前传播foward
    def call(self, x):
        #conv1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        #conv2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        #conv3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        #conv4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        #conv5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        x = self.flatten(x)
        #fc6
        x = self.fc6(x)
        #fc7
        x = self.fc7(x)
        #fc8
        return self.fc8(x)


        # print x

def loadDataSet():
    filePath = "/media/jade/119f84e1-83d3-44cc-98c5-b52551f23158/home/jade/Data/StaticDeepFreeze/sdfgoods10"
    filenames = os.listdir(filePath)
    x_train = []
    y_train = []
    for i in range(len(filenames)):
        imagenames = GetAllImagesPath(os.path.join(filePath,filenames[i]))
        for imagename in imagenames:
            x_train.append(cv2.resize(cv2.imread(imagename),(224,224)).astype("float32")/255.0)
            y_train.append([i])
        print("Loading....")
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    train_random_arr = np.random.randint(0, x_train.shape[0], size= x_train.shape[0])
    x_train = x_train[train_random_arr,:,:,:]
    y_train = y_train[train_random_arr,:]
    return x_train[0:4000,:,:,:],y_train[0:4000,:],x_train[4000:,:,:,:],y_train[4000:,:]




def train():
    x_train,y_train,x_test,y_test = loadDataSet()
    model = VGGNet(classes=10)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                  # loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0)
    model.summary()
    model.save_weights('VGG', save_format='tf')

def predict():
    x_train, y_train, x_test, y_test = loadDataSet()
    # 读取保存的模型参数
    new_model = VGGNet(classes=10)
    # new_model.train_on_batch(x_train[:1], y_train[:1])
    new_model.load_weights('VGG')
    new_predictions = new_model.predict(x_test)
    for i in range(new_predictions.shape[0]):
        # print(new_predictions[i])
        print(np.argmax(new_predictions[i]))
        print(y_test[i])



if __name__ == '__main__':
    train()