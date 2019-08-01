#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/1 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/1  上午11:32 modify by jade
import tensorflow as tf
dataset = tf.data.Dataset.range(5)

iterator = iter(dataset)