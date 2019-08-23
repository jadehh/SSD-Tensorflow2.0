#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/20 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/20  下午5:41 modify by jade
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.engine import Layer
from utils.bounding_box_utils import convert_coordinates

class AnchorBoxes(Layer):
    """
    Prior box 层
    """

    def __init__(self, img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 two_boxes_for_ar1=True,
                 this_steps=None,
                 this_offsets=None,
                 clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 coords="centroids",
                 normalize_coors=False,
                 **kwargs):
        """
        box encoding process
        :param img_height:
        :param img_width:
        :param this_scale:
        :param next_scale:
        :param aspect_ratios:
        :param two_boxes_for_ar1:
        :param this_steps:
        :param this_offsets:
        :param clip_boxes:
        :param variances:
        :param coords:
        :param normalize_coors:
        """
        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError("scale 必须在 [0,1]之间")
        if len(variances) != 4:
            raise ValueError("必须要有四个超参数缩放边界框")
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("超参数的值必须大于0")
        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coors
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)

        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        size = min(self.img_height, self.img_width)

        # 在aspect ratios 中 计算宽和高
        w_h_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                box_height = box_width = self.this_scale * size
                w_h_list.append((box_width, box_height))

                if (self.two_boxes_for_ar1):
                    # 计算通过这个比例值和下一个比例值，计算一个稍大的版本
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    w_h_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                w_h_list.append((box_width, box_height))
        w_h_list = np.array(w_h_list)

        batch_size, feature_map_height, feature_map_width, feature_map_channels = x.shape

        if self.this_steps is None:
            step_height = self.img_height / feature_map_height
            step_width = self.img_width / feature_map_width
        else:
            if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps
        # 计算offsets
        if self.this_offsets is None:
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets

        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height,
                         feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width,
                         feature_map_width)

        cx_grid, cy_grid = np.meshgrid(cx, cy)

        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 2] = w_h_list[:, 0]
        boxes_tensor[:, :, :, 3] = w_h_list[:, 1]

        boxes_tensor = convert_coordinates(boxes_tensor,start_index = 0,conversion="centroids2corners")
        if self.clip_boxes:
            x_coords = boxes_tensor[:,:,:,[0,2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0,2]] = x_coords
            y_coords = boxes_tensor[:,:,:,[1,3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1,3]] = y_coords

        if self.normalize_coords:
            boxes_tensor[:,:,:,[0,2]] /= self.img_width
            boxes_tensor[:,:,:,[1,3]] /= self.img_height

        if self.coords == "centroids":
            boxes_tensor = convert_coordinates(boxes_tensor,start_index=0,conversion="corners2centroids",border_pixel="half")

        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances

        boxes_tensor = np.concatenate((boxes_tensor,variances_tensor),axis=-1)

        boxes_tensor = np.expand_dims(boxes_tensor,axis=0)

        boxes_tensor = K.tile(K.constant(boxes_tensor,dtype="float32"),(K.shape(x)[0],1,1,1,1))

        #shape = (feature_map_height,feature_map_width,n_boxes,8)
        return boxes_tensor

