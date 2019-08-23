#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/20 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/20  下午4:21 modify by jade
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.engine import Layer

#自定义layer层，会有正向传播，和反向传播
class L2Normalization(Layer):
    """
     在深度神经网络中，偶尔会出现多个量纲不同的向量拼接在一起的情况，
     此时就可以使用L2归一化统一拼接后的向量的量纲，使得网络能够快速收敛。
    """
    def __init__(self,gamma_init = 20, **kwargs):
        self.axis = 3
        self.gamma_init = gamma_init
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        gamma = self.gamma_init * np.ones((input_shape[self.axis],))
        self.gamma = K.variable(gamma,name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]
        super(L2Normalization,self).build(input_shape)

    def call(self,x,mask=None):
        outputs = K.l2_normalize(x,self.axis)
        return outputs * self.gamma