#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/20 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/20  下午4:46 modify by jade
import numpy as np
class Anchors():
    def __init__(self,args):
        self.aspect_ratios_global = None
        self.aspect_ratios_per_layer = [[1.0,2.0,0.5], [1.0,2.0,0.5,3.0,1.0/3.0], [1.0,2.0,0.5,3.0,1.0/3.0],
                                        [1.0,2.0,0.5,3.0,1.0/3.0], [1.0,2.0,0.5],[1.0,2.0,0.5]]
        self.two_boxes_for_ar1 = True
        self.steps = [8,16,32,64,100,300]
        self.offsets = [0.5,0.5,0.5,0.5,0.5,0.5]
        self.clip_boxes = False,
        self.variances = [0.1,0.1,0.2,0.2] #训练时候对边界框进行缩放
        self.n_predictor_layers = 6
        self.scales = [0.1,0.2,0.37,0.54,0.71,0.88,1.05]
        # self.min_scale = args.min_scale
        # self.max_scale = args.max_scale
        # self.num_classes = args.num_classes + 1

    def layer_boxes(self):
        if self.scales:
            if len(self.scales) != self.n_predictor_layers + 1:
                raise ValueError("必须等于特征层的数目")
        # else:
        #     self.scales = np.linspace(self.min_scale,self.max_scale,self.n_predictor_layers)

        if len(self.variances) != 4:
            raise ValueError("4 个参数是必须的")

        self.variances = np.array(self.variances)
        if np.any(self.variances <= 0):
            raise ValueError("超参数的值必须大于0")

        if (not (self.steps is None)) and (len(self.steps)!=self.n_predictor_layers):
            raise ValueError("steps 的步数必须等于特征图的个数")

        if (not (self.offsets is None)) and (len(self.offsets) != self.n_predictor_layers):
            raise ValueError("offset的个数必须等于特征图个数")

        ############################################
        # 计算 anchor box 的参数
        ############################################

        if self.aspect_ratios_per_layer:
            aspect_ratios = self.aspect_ratios_per_layer
        else:
            aspect_ratios = [self.aspect_ratios_global] * self.n_predictor_layers

        if self.aspect_ratios_per_layer:
            n_boxes = []
            for ar in self.aspect_ratios_per_layer:
                if (1 in ar) and self.two_boxes_for_ar1:
                    n_boxes.append(len(ar) + 1)
                else:
                    n_boxes.append(len(ar))
        else:
            if (1 in self.aspect_ratios_global) and self.two_boxes_for_ar1:
                n_boxes = len(self.aspect_ratios_global) + 1
            else:
                n_boxes = len(self.aspect_ratios_global)

            n_boxes = [n_boxes] * self.n_predictor_layers

        if self.steps is None:
            self.steps = [None] * self.n_predictor_layers
        if self.offsets is None:
            self.offsets = [None] * self.n_predictor_layers

        return n_boxes
