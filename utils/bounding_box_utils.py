#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/21 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/21  上午10:09 modify by jade

import numpy as np
def convert_coordinates(tensor,start_index,conversion,border_pixel='half'):
    """
    两种坐标格式的转换
    :param tensor:
    :param start_index:
    :param conversion:
    :param border_pixel:
    :return:
    """

    if border_pixel == 'half':
        d = 0
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+2]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+1] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+2] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+1] + d # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+2] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    return tensor1