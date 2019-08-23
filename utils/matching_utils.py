#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/23 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/23  下午5:43 modify by jade

import numpy as np

def match_bipartite_greedy(weight_martix):
    """
    找到框的 anchor 与 真实框 IOU 最大的那个
    :param weight_martix:
    :return:
    """

    weight_martix = np.concatenate(weight_martix)

    num_ground_truth_boxes = weight_martix.shape[0]  #真实标签目标框个数

    all_gt_indices  = list(range(num_ground_truth_boxes))

    matches = np.zeros(num_ground_truth_boxes,dtype=np.int)

    for _ in range(num_ground_truth_boxes):

        anchor_indices = np.argmax(weight_martix,axis=1) ##找到真实框与AnchorIOU最大值的索引
        #当grount_truth_box 数量 大于 1 ，可能会有多个值
        #anchor_indicaes shape = [num_ground_truth_boxes]
        overlaps = weight_martix[all_gt_indices,anchor_indices] ##这是最大值

        #在最大值列表里面挑一个最大的值索引
        grount_truth_index = np.argmax(overlaps)

        anchor_index = anchor_indices[grount_truth_index]

        matches[grount_truth_index] = anchor_indices

        weight_martix[grount_truth_index] == 0  #将那一行都为0
        weight_martix[:,anchor_index] = 0  # 将所有anchor_index 所有值都为0


    return matches



def match_multi(weight_matrix,threshold):
    """
    匹配多个Anchor
    :param weight_matrix:
    :param threshold:
    :return:
    """
    num_anchor_boxes = weight_matrix.shape[1]
    all_anchor_indices = list(range(num_anchor_boxes))

    #找到最佳匹配的

    grount_truth_indices = np.argmax(weight_matrix,axis=0)
    overlaps = weight_matrix[grount_truth_indices,all_anchor_indices]
    #overlaps 代表的是weight_matrix里面所有的值

    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]
    gt_indices_thresh_met = grount_truth_indices[anchor_indices_thresh_met]

    return gt_indices_thresh_met,anchor_indices_thresh_met  ## 前面一个参数表示哪一行，后面表示那一列
