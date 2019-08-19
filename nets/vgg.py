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

class VGGNet16(Model):
    def __init__(self, classes=10):
        super(VGGNet16, self).__init__()
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


def LoadTFRecord(tfrecord_path, batch_size=32, shuffle=True,repeat=True,is_train=True):
    return loadClassifyTFRecord(tfrecord_path, batch_size, shuffle,repeat,is_train=is_train)

def createTFRecord(path,datasetname,proto_txt_path):
    CreateClassifyTFRecorder(path,datasetname,proto_txt_path)

def train(train_tfrecord_path,test_tfrecord_path,num_classes=10):
    train_batch_generator = LoadTFRecord(train_tfrecord_path,batch_size=8)
    test_batch_generator = LoadTFRecord(test_tfrecord_path,batch_size=8)


    # x_train,y_train,x_test,y_test = loadDataSet()
    model = VGGNet16(classes=num_classes)
    # # model.load_weights("VGGNetDense")
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    #
    history = model.fit_generator(generator=train_batch_generator,
                                  steps_per_epoch=10,
                                  epochs=10,
                                  verbose=1,
                                  validation_data=test_batch_generator,
                                  validation_steps=1)
    model.summary()
    model.save_weights('VGGNetDense', save_format='tf')


def predict():
    # 读取保存的模型参数

    new_model = VGGNet16(classes=20)
    # new_model.train_on_batch(x_train[:1], y_train[:1])
    new_model.load_weights('checkpoints/voc_vgg16net')
    test_generator = loadClassifyTFRecord("/home/jade/Data/VOCdevkit/TFRecords/VOC_train.tfrecord",repeat=False)
    num = 0
    correct = 0
    for test_images, test_labels in test_generator:
        new_predictions = new_model.predict(test_images)
        for i in range(new_predictions.shape[0]):
            num = num + 1
            print(np.argmax(new_predictions[i]),test_labels[i].numpy())
            if np.argmax(new_predictions[i]) == test_labels[i].numpy():
                correct = correct + 1
    print(correct/float(num))


def train2(train_path,test_path,num_classes=10,datasetname='voc'):

    train_num = int(len(GetLabelAndImagePath("/home/jade/Data/VOCdevkit/Classify")) * 0.8)
    test_num = int(len(GetLabelAndImagePath("/home/jade/Data/VOCdevkit/Classify")) * 0.2)
    batch_size = 32
    model = VGGNet16(classes=num_classes)
    model.load_weights("checkpoints/voc_vgg16net_2019-08-15")


    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam(0.00001)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 50
    print("start training ....")
    for epoch in range(EPOCHS):
        train_ds = LoadTFRecord(train_path, batch_size=batch_size, repeat=False,is_train=True)
        test_ds = LoadTFRecord(test_path, batch_size=batch_size, repeat=False,is_train=False)
        train_processbar = ProcessBar()
        train_processbar.count = int(train_num / batch_size)
        for images, labels in train_ds:
            train_processbar.start_time = time.time()
            train_step(images, labels)
            template = 'Training Epoch: {} || Loss: {} || Accuracy: {}%'
            NoLinePrint(template.format(epoch + 1,
                              format(train_loss.result(),'.2f'),
                              format(train_accuracy.result()* 100, '.2f')),train_processbar)

        print("")
        if (epoch+1) % 10 == 0 and epoch !=0:
            test_processbar = ProcessBar()
            test_processbar.count = int(test_num / batch_size)
            for test_images, test_labels in test_ds:
                test_step(test_images, test_labels)
                test_processbar.start_time = time.time()
                template = 'Testing  Epoch: {} || Loss: {} || Accuracy: {}%'
                NoLinePrint(template.format(epoch + 1,
                                            format(test_loss.result(), '.2f'),
                                            format(test_accuracy.result() * 100, '.2f')), test_processbar)

            print("")

    CreateSavePath("checkpoints")
    model.summary()
    model.save_weights('checkpoints/'+datasetname+'_vgg16net'+"_"+GetToday(), save_format='tf')





if __name__ == '__main__':
    train2("/home/jade/Data/VOCdevkit/TFRecords/VOC_train.tfrecord","/home/jade/Data/VOCdevkit/TFRecords/VOC_test.tfrecord",20)
    # train_ds = LoadTFRecord("/home/jade/Data/VOCdevkit/TFRecords/VOC_train.tfrecord",repeat=False)
    # for i in range(10):
    #     for images, labels in train_ds:
    #         print("Load dataset ...")


    # train2()
    # predict()
    #ResizeClassifyDataset("/home/jade/Data/VOCdevkit/Classify",224)
    #VOCTOClassify("/media/jade/119f84e1-83d3-44cc-98c5-b52551f23158/home/jade/Data/VOCdevkit/VOC2012")
    #XMLTOPROTXT("/home/jade/Data/VOCdevkit/voc.xlsx","VOC")
    #createTFRecord("/home/jade/Data/VOCdevkit/Classify_resize","VOC","/home/jade/Data/VOCdevkit/VOC.prototxt")
    #train("/home/jade/Data/sdfgoods/TFRecords/sdfgoods10_224_train.tfrecord","/home/jade/Data/sdfgoods/TFRecords/sdfgoods10_224_test.tfrecord",num_classes=10)

    #predict()
