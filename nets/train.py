#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/16 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/16  上午9:27 modify by jade
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.applications import VGG16
from nets.vgg import VGGNet16,VGGNetConv
from jade import *
from datasetopeation.jadeClassifyTFRecords import LoadClassifyTFRecord
class TrainModel():
    def __init__(self,args):
        self.checkpoint_path = args.checkpoint_path
        self.model_name = args.model_name
        self.train_path = args.train_path
        self.test_path = args.test_path
        self.batch_size = args.batch_size
        self.classify_path = args.classify_path
        self.num_classes = args.num_classes
        self.dataset_name = args.dataset_name
        self.train_num = int(len(GetLabelAndImagePath(args.classify_path)) * 0.9 / self.batch_size)
        self.test_num = int(len(GetLabelAndImagePath(args.classify_path)) * 0.1 / self.batch_size)
        self.epochs = 50
        self.learning_rate = 0.1
        self.init_net()
        self.init_loss()

    def init_net(self):
        if self.model_name == "vgg":
            self.model = VGGNetConv(self.num_classes)
            self.model.build(input_shape=(self.batch_size,224,224,3))
        # if self.checkpoint_path:
        #     self.model.load_weights(self.checkpoint_path)

    def init_loss(self):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

        self.optimizer = tf.keras.optimizers.SGD()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def train_step(self,images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    def test_step(self,images, labels):
        predictions = self.model(images)
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)


    def loadData(self,repeat=False):
        train_ds = LoadClassifyTFRecord(self.train_path, self.batch_size, shuffle=True, repeat=repeat, is_train=True)
        test_ds = LoadClassifyTFRecord(self.test_path,self.batch_size,shuffle=True,repeat=repeat,is_train=False)
        return train_ds,test_ds


    def train(self):
        train_ds,test_ds = self.loadData()

        print("start training ....")
        for epoch in range(self.epochs):
            train_processbar = ProcessBar()
            train_processbar.count = self.train_num
            for images, labels in train_ds:
                K.set_value(self.optimizer.lr,self.learning_rate)
                train_processbar.start_time = time.time()
                self.train_step(images, labels)
                template = 'Training Epoch: {} || learning rate: {} || Loss: {} || Accuracy: {}%'
                NoLinePrint(template.format(epoch + 1,
                                            format(self.optimizer.lr.numpy(), '.5f'),
                                            format(self.train_loss.result(), '.2f'),
                                            format(self.train_accuracy.result() * 100, '.2f')), train_processbar)

            print("")
            test_processbar = ProcessBar()
            test_processbar.count = self.test_num
            for test_images, test_labels in test_ds:
                test_processbar.start_time = time.time()
                self.test_step(test_images, test_labels)

                template = 'Testing  Epoch: {} || Loss: {} || Accuracy: {}%'
                NoLinePrint(template.format(epoch + 1,
                                                format(self.test_loss.result(), '.2f'),
                                                format(self.test_accuracy.result() * 100, '.2f')), test_processbar)
            print("")

        save_path = CreateSavePath("/home/jade/Models/"+self.dataset_name+"Classify/")
        self.model.summary()
        self.model.save_weights(save_path + self.dataset_name + '_vgg16net' + "_" + GetToday(), save_format='tf')


    def evaluate(self):
        test_ds = LoadClassifyTFRecord(self.test_path,self.batch_size,shuffle=True,repeat=False,is_train=False)

        test_processbar = ProcessBar()
        test_processbar.count = self.test_num
        for test_images, test_labels in test_ds:
            test_processbar.start_time = time.time()
            self.test_step(test_images, test_labels)

            template = 'Testing || Loss: {} || Accuracy: {}%'
            NoLinePrint(template.format(format(self.test_loss.result(), '.2f'),
                                        format(self.test_accuracy.result() * 100, '.2f')), test_processbar)
    def lr_schedule(self,epoch):
        if epoch < 10:
            return 0.001
        elif epoch < 50:
            return 0.0001
        else:
            return 0.00001

    def keras_train(self):
        train_ds,test_ds = self.loadData(repeat=True)
        self.model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        learning_rate_schedule = LearningRateScheduler(schedule=self.lr_schedule,verbose=1)
        callbacks = [learning_rate_schedule]
        #
        # history = self.model.fit_generator(generator=train_ds,
        #                               steps_per_epoch=self.train_num,
        #                               epochs=self.epochs,
        #                               validation_data=test_ds,
        #                               validation_steps=self.test_num,
        #                               callbacks=callbacks,
        #                               initial_epoch=0)
        self.model.fit(train_ds,
                       test_ds,
                       batch_size=self.batch_size,
                       callbacks=callbacks,
                       shuffle=True,
                       epochs=self.epochs)
        self.model.summary()
        save_path = CreateSavePath("/home/jade/Models/"+self.dataset_name+"Classify")
        self.model.save_weights(os.path.join(save_path, self.dataset_name + '_vgg16net' + "_" + GetToday()+".h5"))

if __name__ == '__main__':
    import argparse

    paraser = argparse.ArgumentParser(description="VOC Classify")
    # genearl
    paraser.add_argument("--checkpoint_path",
                         default="/home/jade/Checkpoints/VGG/vgg16_weights_tf_dim_ordering_tf_kernels.h5",
                         help="path to load model")
    paraser.add_argument("--model_name", default="vgg", help="model_name")
    paraser.add_argument("--train_path", default="/home/jade/Data/sdfgoods/TFRecords/sdfgoods_train.tfrecord", help="train_path")
    paraser.add_argument("--test_path", default="/home/jade/Data/sdfgoods/TFRecords/sdfgoods_test.tfrecord", help="test_path")
    paraser.add_argument("--batch_size", default=32, help="batch_size")
    paraser.add_argument("--classify_path", default="/home/jade/Data/sdfgoods/sdfgoods10", help="classify_path")
    paraser.add_argument("--num_classes", default=10, help="num_classes")
    paraser.add_argument("--dataset_name", default="sdfgoods", help="dataset_name")

    args = paraser.parse_args()

    trainModel = TrainModel(args)
    trainModel.train()