#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 作者：2019/8/21 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/21  上午11:12 modify by jade
from ssd512 import SSD512
import tensorflow as tf
from keras_loss_function.keras_ssd_loss import SSDLoss
from tensorflow.python.keras.callbacks import LearningRateScheduler
from datasetopeation.jadeVocTFRecord import LoadVOCTFRecord
from utils.ssd_input_encode import SSDInputEncoded
import numpy as np




class Train():
    def __init__(self,args):
        self.args = args
        self.batch_size = args.batch_size
        self.train_path = args.train_path
        self.test_path = args.test_path
        self.epochs = 100
        self.init_net()
        self.encode()

    def init_net(self):
        self.model = SSD512(self.args)
        self.model.build(input_shape=(self.batch_size, 300, 300, 3))

    def init_loss(self):
        sgd = tf.keras.optimizers.SGD(lr=0.001,momentum=0.9,decay=0.0,nesterov=False)
        ssd_loss = SSDLoss(neg_pos_ratio=0.3,alpha=1)
        self.model.compile(optimizer=sgd,loss=ssd_loss.compute_loss)
    def lr_schedule(self,epoch):
        if epoch < 10:
            return 0.001
        elif epoch < 50:
            return 0.0001
        else:
            return 0.00001

    def loadData(self,repeat=False):
        train_ds = LoadVOCTFRecord(self.train_path, 1, shuffle=False, repeat=repeat, is_train=True)
        test_ds = LoadVOCTFRecord(self.test_path,1,shuffle=False,repeat=repeat,is_train=False)
        return train_ds,test_ds
    def encode(self):
        predictor_sizes = [(38,38),(19,19),(10,10),(5,5),(3,3),(1,1)]
        self.ssdEncode = SSDInputEncoded(self.args.image_size[0],self.args.image_size[1],self.args.num_classes,
                                         predictor_sizes,aspect_ratios_per_layer=self.args.aspect_ratios,
                                         steps=self.args.this_steps,offsets=self.args.this_offsets)
    def train(self):

        train_ds, test_ds = self.loadData(repeat=False)
        learning_rate_schedule = LearningRateScheduler(schedule=self.lr_schedule, verbose=1)
        callbacks = [learning_rate_schedule]


        train_dataset = []
        for (image,label) in train_ds:
            label_np = label.numpy()
            encode_label = self.ssdEncode.encode(label.numpy())
            train_dataset.append((image,encode_label))

        train_dataset = np.array(train_dataset)
        self.model.fit(train_dataset,
                       callbacks=callbacks,
                       shuffle=True,
                       epochs=self.epochs)

if __name__ == '__main__':
    import argparse
    paraser = argparse.ArgumentParser(description="SSD")
    paraser.add_argument("--num_classes", default=2,help="num_classes")
    paraser.add_argument("--image_size", default=[300,300], help="image_size")
    paraser.add_argument("--this_scale", default=[0.07,0.15,0.33,0.51,0.69,0.87,1.05], help="this_scale")
    paraser.add_argument("--aspect_ratios", default=[[1.0,2.0,0.5], [1.0,2.0,0.5,3.0,1.0/3.0], [1.0,2.0,0.5,3.0,1.0/3.0],
                                        [1.0,2.0,0.5,3.0,1.0/3.0], [1.0,2.0,0.5],[1.0,2.0,0.5]], help="aspect_ratios")
    paraser.add_argument("--two_boxes_for_ar1", default=True, help="two_boxes_for_ar1")
    paraser.add_argument("--this_steps", default=[8,16,32,64,100,300], help="this_steps")
    paraser.add_argument("--this_offsets", default= [0.5,0.5,0.5,0.5,0.5,0.5], help="this_offsets")
    paraser.add_argument("--clip_boxes", default=False, help="clip_boxes")
    paraser.add_argument("--variances", default= [0.1,0.1,0.2,0.2], help="variances")
    paraser.add_argument("--coords", default="centroids", help="coords")
    paraser.add_argument("--normalize_coors", default=True, help="normalize_coors")
    paraser.add_argument("--train_path", default=r"F:\Data\HandGesture\TFRecords/UA_Handgesture_test.tfrecord",
                         help="train_path")
    paraser.add_argument("--test_path", default= r"F:\Data\HandGesture\TFRecords/UA_Handgesture_test.tfrecord",
                         help="test_path")
    paraser.add_argument("--batch_size", default=32, help="batch_size")
    args = paraser.parse_args()

    train = Train(args)
    train.train()

