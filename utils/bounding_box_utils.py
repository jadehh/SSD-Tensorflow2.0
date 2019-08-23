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

def iou(boxes1,boxes2,coords='centroids',mode='out_product',border_pixels='half'):
    """
    IOU 对比，同一张图里面的所有目标框与所有的anchor做IOU对比
    :param boxes1:
    :param boxes2:
    :param coords:
    :param mode:
    :param border_pixesl:
    :return:
    """

    if boxes1.ndim >2:
        raise ValueError("boxes1 的 shape 为 1或者2,不同为其他的")
    if boxes2.ndim > 2:
        raise ValueError("boxes2 的 shape 为 1或者2,不同为其他的")

    if boxes1.ndim == 1:boxes1 = np.expand_dims(boxes1,axis=0)
    if boxes2.ndim == 1:boxes2 = np.expand_dims(boxes2,axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("框里面的坐标必须为4个")
    if not mode in {"outer_product","element-wise"}:
        raise ValueError("mode 必须为 'outer_product','element-wise这两种'")

    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1,start_index=0,conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2,start_index=0,conversion='centroids2corners')
        coords = 'corners'
    elif not (coords in {'minmax','corners'}):
        raise ValueError ("coords 必须在minmax,corners 里面")


    ##计算IOU

    intersection_areas = intersection_area_(boxes1,boxes2,coords=coords,mode=mode)
    m = boxes1.shape[0]
    n = boxes2.shape[0]

    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3
    if border_pixels == 'half':
        d = 0

    if mode == 'outer_product':
        boxes1_areas = np.tile(np.expand_dims((boxes1[:,xmax]-boxes1[:,xmin] +d ) * (boxes1[:,ymax]- boxes1[:,ymin] +d ),axis=1),reps=(1,n))

        boxes2_areas = np.tile(np.expand_dims((boxes2[:,xmax]-boxes2[:,xmin] +d ) * (boxes2[:,ymax]- boxes2[:,ymin] +d ),axis=0),reps=(m,1))


    unio_ares = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / unio_ares

def intersection_area_(boxes1,boxes2,coords,mode,border_pixels='half'):
    """

    :param boxes1:
    :param boxes2:
    :param coords:
    :param mode:
    :param border_pixels:
    :return:
    """
    m = boxes1.shape[0]
    n = boxes2.shape[0]

    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1



    if mode=="outer_product":
        #保持两个boxes维度一样 (m*n*4)


        #在小的里面找最大的 此时维度变成(m*n*2)
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:,[xmin,ymin]],axis=1),reps=(1,n,1)),
                            np.tile(np.expand_dims(boxes2[:,[xmin,ymin]],axis=0),reps=(m,1,1)))

        #在大的里面找最小的 此时维度变成(m*n*2)
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:,[xmax,ymax]],axis=1),reps=(1,n,1)),
                            np.tile(np.expand_dims(boxes2[:,[xmax,ymax]],axis=0),reps=(m,1,1)))

        #计算相交矩形的边长,如果大于0就说明相交，小于0就没有相交
        side_lengths = np.maximum(0,max_xy-min_xy+d)
        return side_lengths[:,:,0]*side_lengths[:,:,1]