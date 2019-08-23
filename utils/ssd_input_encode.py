#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/22 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/22  下午6:42 modify by jade
import numpy as np
from utils.bounding_box_utils import convert_coordinates,iou
from utils.matching_utils import match_bipartite_greedy,match_multi
class SSDInputEncoded:
    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 min_scale=0.1,
                 max_scale=0.9,
                 scales=None,
                 aspect_ratios_global=(0.5,1.0,2.0),
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1 = True,
                 steps=None,
                 offsets=None,
                 clip_boxes=False,
                 variances=[0.1,0.1,0.2,0.2],
                 matching_type='multi',
                 pos_iou_threshold=0.5,
                 neg_iou_limit=0.3,
                 border_pixels='half',
                 coords='centroids',
                 normalize_coors=True,
                 background_id=0):
        """

        将输入encode成[batch,8732,num_classes+predict_boxes+anchors+variances]
        :param img_height:
        :param img_width:
        :param n_classes:
        :param predictor_sizes:
        :param min_scale:
        :param max_scale:
        :param scales:
        :param aspect_ratios_global:
        :param aspect_ratios_per_layer:
        :param two_boxes_for_ar1:
        :param steps:
        :param offsets:
        :param clip_boxes:
        :param variances:
        :param matching_type:
        :param pos_iou_threshold:
        :param neg_iou_limit:
        :param border_pixels:
        :param coords:
        :param normalize_coors:
        :param background_id:
        """

        predictor_sizes = np.array(predictor_sizes)
        #####################################################
        ######## 异常情况判断
        #####################################################

        if predictor_sizes.ndim == 1:
            predictor_sizes = np.expand_dims(predictor_sizes,axis=0)
        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("min_scale max_scale scales 不能为空")

        if scales:
            if (len(scales)!=predictor_sizes.shape[0]+1):
                raise ValueError("scales 必须等于predicor_size.shape[0] + 1")
            scales = np.array(scales)
            if np.any(scales < 0):
                raise ValueError ("scales 里面的数值必须大于0 ")
        else:
            if not 0<min_scale <=max_scale:
                raise ValueError("最大值必须要比最小值要大")

        if not (aspect_ratios_per_layer is None):
            if len(aspect_ratios_per_layer) != predictor_sizes.shape[0]:
                raise ValueError("aspect_ratios_per_layer的长度 必须等于 predictor_sizes的长度")
            for aspect_ratios in aspect_ratios_per_layer:
                if np.any(np.array(aspect_ratios) <= 0):
                    raise ValueError("aspect_ratios_per_layer 里面不能有小于等于0 的数")

        else:
            if aspect_ratios_global is None:
                raise ValueError("aspect_ratios_per_layer 和 aspect_ratios_global 必须有一个不为空")
            if np.any(np.array(aspect_ratios_global) <=0 ):
                raise ValueError("aspect_ratios_global 里面不能有小于0的数")


        if len(variances) !=4:
            raise ValueError("variances 必须要是4个")
        variances = np.array(variances)
        if np.any(variances <=0 ):
            raise ValueError("variances 中不能有小于0的数")

        if not(coords == "minmax" or coords == "centroids" or coords == "corners"):
            raise ValueError("coords 不在规则当中")

        if (not (steps is None)) and (len(steps)!=predictor_sizes.shape[0]):
            raise ValueError("必须为每一个预测层提供一个step")

        if (not (offsets is None)) and (len(offsets) != predictor_sizes.shape[0]):
            raise ValueError("必须为每一个预测层提供一个offset")

        ####################################################
        # 设置并且计算
        ####################################################

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes + 1
        self.predictor_size = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        if scales is None:
            self.scales = np.linspace(self.min_scale,self.max_scale,len(predictor_sizes) + 1)
        else:
            self.scales = scales

        if aspect_ratios_per_layer is None:
            self.aspect_ratios = [aspect_ratios_global]*predictor_sizes.shape[0]
        else:
            self.aspect_ratios = aspect_ratios_per_layer

        self.two_boxes_for_ar1 = two_boxes_for_ar1
        if not (steps is None):
            self.steps = steps
        else:
            self.steps = [None] * predictor_sizes.shape[0]
        if not (offsets is None):
            self.offsets = offsets
        else:
            self.offsets = [None] * predictor_sizes.shape[0]

        self.clip_boxes = clip_boxes
        self.variances = variances
        self.matching_type = matching_type
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.border_pixels = border_pixels
        self.coords = coords
        self.normalize_coors = normalize_coors
        self.background_id = background_id

        #计算每个预测层的框数量，比如 38*38 应该是4个框
        if not (aspect_ratios_per_layer is None):
            self.n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                if (1 in aspect_ratios) & two_boxes_for_ar1:
                    self.n_boxes.append(len(aspect_ratios) + 1)
                else:
                    self.n_boxes.append(len(aspect_ratios))
        else:
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                self.n_boxes = len(aspect_ratios_global)
            else:
                self.n_boxes = len(aspect_ratios_global)



        #####################################################################
        ######## 计算每个预测层的Anchor
        #####################################################################

        self.box_list = []
        self.wh_list_diag = []
        self.steps_diag = []
        self.offsets_diag = []
        self.centers_diag = []

        for i in range(len(self.predictor_size)):
            boxes,center,wh,step,offset = self.generate_anchor_boxes_for_layer(feature_maps_size=self.predictor_size[i],
                                                                               aspect_ratios=self.aspect_ratios[i],
                                                                               this_scale=self.scales[i],
                                                                               next_scale=self.scales[i+1],
                                                                               this_steps=self.steps[i],
                                                                               this_offsets=self.offsets[i],
                                                                               diagnostics=True)

            self.box_list.append(boxes)
            self.wh_list_diag.append(wh)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)
            self.centers_diag.append(center)


    def encode(self,ground_truth_labels,diagnostics=False):
        """
        将真实坐标 [class_id,xmin,ymin,xmax,ymax] 映射到每个anchor上 [8732,num_classes+predicors+anchors+variances]
        :param ground_truth_labels:
        :param diagnostics:
        :return:
        """
        #下面代表的是每个参数的索引
        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        batch_size = len(ground_truth_labels)

        #############################################
        ### 生成y_encode
        #############################################

        y_encoded = self.generate_encoding_template(batch_size=batch_size,diagnostics=False)

        ###############################################
        ### 与groundth 做 匹配
        ###############################################

        y_encoded[:,:,self.background_id] = 1  ## 默认所有类别都是background
        b_boxes = y_encoded.shape[1]
        class_vectors = np.eye(self.n_classes) #one hot [0] * self.num_classes

        for i in range(batch_size):
            if ground_truth_labels[i].size == 0:
                #表示这个图里面没有目标
                continue
            labels = ground_truth_labels[i].astype(np.float)

            if np.any(labels[:,[xmax]]-labels[:,[xmin]] <= 0) or (np.any(labels[:,[ymax]]-labels[:,[ymin]]) <=0):
                raise ValueError ("这个数据有问题")

            if self.normalize_coors:
                labels[:,[ymin,ymax]] /=self.img_height
                labels[:,[xmin,xmax]] /= self.img_width
            if self.coords == "centroids":
                labels = convert_coordinates(labels,start_index=xmin,conversion="corners2centroids",border_pixel=self.border_pixels)
            elif self.coords == 'minmax':
                labels = convert_coordinates(labels,start_index=min,conversion='corners2minmax')

            class_one_hot = class_vectors[labels[:,class_id].astype(np.int)]
            labels_one_hot = np.concatenate([class_one_hot,labels[:,[xmin,ymin,xmax,ymax]]],axis=-1)

            #[[x1,y1,x2,y2],[x3,y3,x4,y4]] 与 所有的anchor 框做IOU对比
            similarities = iou(labels[:,[xmin,ymin,xmax,ymax]],y_encoded[i,:,-12:-8],coords=self.coords,mode='outer_product',border_pixels=self.border_pixels)

            bipartite_matches = match_bipartite_greedy(weight_martix=similarities)

            #IOU值最大的就是真实数据

            y_encoded[i,bipartite_matches, :-8] = labels_one_hot

            similarities[:,similarities] = 0 #计算过后置位0，下一次不参与计算

            if self.matching_type == 'multi':
                mathes = match_multi(similarities)
                y_encoded[i, mathes[1], :-8] = labels_one_hot[mathes[0]] ##多个目标框

                similarities[:,mathes[1]] = 0


            #IOU大于0.3 且 小于0.5 既不是背景 也不是前景，这些数据的类别全是0
            max_background_similarties = np.amax(similarities,axis=0)  #返回最大的值
            neutral_boxes = np.nonzero(max_background_similarties >= self.neg_iou_limit)[0]
            y_encoded[i,neutral_boxes,self.background_id] = 0

        if self.coords == 'centroids':
            y_encoded[:,:,[-12,-11]] -= y_encoded[:,:,[-8,-7]] # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            y_encoded[:,:,[-12,-11]] /= y_encoded[:,:,[-6,-5]] * y_encoded[:,:,[-4,-3]] # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
            y_encoded[:,:,[-10,-9]] /= y_encoded[:,:,[-6,-5]] # w(gt) / w(anchor), h(gt) / h(anchor)
            y_encoded[:,:,[-10,-9]] = np.log(y_encoded[:,:,[-10,-9]]) / y_encoded[:,:,[-2,-1]] # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
        elif self.coords == 'corners':
            y_encoded[:,:,-12:-8] -= y_encoded[:,:,-8:-4] # (gt - anchor) for all four coordinates
            y_encoded[:,:,[-12,-10]] /= np.expand_dims(y_encoded[:,:,-6] - y_encoded[:,:,-8], axis=-1) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:,:,[-11,-9]] /= np.expand_dims(y_encoded[:,:,-5] - y_encoded[:,:,-7], axis=-1) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:,:,-12:-8] /= y_encoded[:,:,-4:] # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively
        elif self.coords == 'minmax':
            y_encoded[:,:,-12:-8] -= y_encoded[:,:,-8:-4] # (gt - anchor) for all four coordinates
            y_encoded[:,:,[-12,-11]] /= np.expand_dims(y_encoded[:,:,-7] - y_encoded[:,:,-8], axis=-1) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:,:,[-10,-9]] /= np.expand_dims(y_encoded[:,:,-5] - y_encoded[:,:,-6], axis=-1) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:,:,-12:-8] /= y_encoded[:,:,-4:] # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively

        if diagnostics:
            # Here we'll save the matched anchor boxes (i.e. anchor boxes that were matched to a ground truth box, but keeping the anchor box coordinates).
            y_matched_anchors = np.copy(y_encoded)
            y_matched_anchors[:,:,-12:-8] = 0 # Keeping the anchor box coordinates means setting the offsets to zero.
            return y_encoded, y_matched_anchors
        else:
            return y_encoded










    def generate_anchor_boxes_for_layer(self,
                                        feature_maps_size,
                                        aspect_ratios,
                                        this_scale,
                                        next_scale,
                                        this_steps,
                                        this_offsets,
                                        diagnostics=False):
        """
        不同预测层上的框的数量
        :param feature_maps_size:
        :param aspect_ratios:
        :param this_scale:
        :param next_scale:
        :param this_steps:
        :param this_offsets:
        :param diagnostics:
        :return:
        """
        size = min(self.img_height,self.img_width)
        #计算不同尺寸是，box的高和宽
        wh_list = []
        for ar in aspect_ratios:
            if ar == 1:
                box_height = box_width = this_scale * size
                wh_list.append((box_width,box_height))
                if self.two_boxes_for_ar1:
                    box_height = box_width = np.sqrt(this_scale*next_scale) * size
                    wh_list.append((box_width,box_height))
            else:
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width,box_height))

        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        if (this_steps is None):
            step_height = self.img_height / feature_maps_size[0]
            step_width = self.img_width / feature_maps_size[0]
        else:
            if isinstance(this_steps,(list,tuple) and len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps,(int,float)):
                step_height = this_steps
                step_width = this_steps

        if this_offsets is None:
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets,(list,tuple) and len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets,(int,float)):
                offset_height = this_offsets
                offset_width = this_offsets



        cy = np.linspace(offset_height * step_height,(offset_height + feature_maps_size[0] - 1)*step_height,feature_maps_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_maps_size[1] - 1) * step_width, feature_maps_size[1])

        cx_grid,cy_grid = np.meshgrid(cx,cy)
        cx_grid = np.expand_dims(cx_grid,-1)
        cy_grid = np.expand_dims(cy_grid,-1)


        box_tensor = np.zeros((feature_maps_size[0],feature_maps_size[1],n_boxes,4))

        box_tensor[:,:,:,0] = np.tile(cx_grid,(1,1,n_boxes))
        box_tensor[:,:,:,1] = np.tile(cy_grid,(1,1,n_boxes))
        box_tensor[:,:,:,2] = wh_list[:,0]
        box_tensor[:,:,:,3] = wh_list[:,1]


        if self.normalize_coors:
            box_tensor[:,:,:,[0,2]] /= self.img_width
            box_tensor[:,:,:,[1,3]] /= self.img_height

        if self.coords == "centroids":
            box_tensor = convert_coordinates(box_tensor,start_index=0,conversion="corners2centroids",border_pixel='half')
        elif self.coords == "minmax":
            box_tensor = convert_coordinates(box_tensor,start_index=0,conversion='corners2minmax',border_pixel='half')


        if diagnostics:
            return box_tensor,(cy,cx),wh_list,(step_height,step_width),(offset_height,offset_width)
        else:
            return box_tensor

    def generate_encoding_template(self,batch_size,diagnostics):
        boxes_batch = []
        for boxes in self.box_list:
            boxes = np.expand_dims(boxes,axis=0)
            boxes = np.tile(boxes,(batch_size,1,1,1,1))

            boxes = np.reshape(boxes,(batch_size,-1,4))
            boxes_batch.append(boxes)

        boxes_tensor = np.concatenate(boxes_batch,axis=1)
        # boxes_tensor 的 shape 为 [batch_size,8732,4]
        class_tensor = np.zeros((batch_size,boxes_tensor.shape[1],self.n_classes))

        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances


        y_encoding_template = np.concatenate((class_tensor,boxes_tensor,variances_tensor),axis=2)

        if diagnostics:
            return y_encoding_template,self.centers_diag,self.wh_list_diag,self.steps_diag,self.offsets_diag
        else:
            return y_encoding_template