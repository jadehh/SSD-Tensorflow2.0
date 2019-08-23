#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/22 by jade
# 邮箱：jadehh@live.com
# 描述：TODO ssd predictor decode
# 最近修改：2019/8/22  下午4:06 modify by jade
from jade import *


class SSDDecoder():
    def __init__(self, proto_txt_path="/home/jade/Data/VOCdevkit/VOC.prototxt"):
        self.categorties, self.classes = ReadProTxt(proto_txt_path)
        self.num_classes = len(self.classes)

    # 编码公式
    """
    cx(predict) = (cx(gt) - cx(anchor)) / w(anchor) / cx_variance
    cy(predict) = (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
    cw(predict)= ln(w(gt) / w(anchor)) / w_variance
    ch(predict) = ln(h(gt) / h(anchor)) / h_variance
    """

    # 解码公式
    """
    cx(gt) = cx(predict) * cx_variance * w(anchor) + cx(anchor)
    cy(gt) = cy(predict) * cy_variance * h(anchor) + cy(anchor)
    
    w(gt) = exp(cw(predict)*w_variance)*w(anchor)
    h(gt) = exp(ch(predict)*h_variance)*h(anchor)
    """

    def decode(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(label.shape[0]):
            class_id = np.argmax(label[i, 0:21])
            predic_bboxes = label[i, 21:25]
            anchor_bboxes = label[i, 25:29]
            #print(predic_bboxes)
            #print(anchor_bboxes)
            variances = label[i, 29:33]
            gt_bboxes_xy = predic_bboxes[0:2] * variances[0:2] * anchor_bboxes[2:4] + anchor_bboxes[0:2]
            gt_bboxes_wh = np.exp(predic_bboxes[2:4] * variances[2:4]) * anchor_bboxes[2:4]
            gt_bboxes = np.concatenate([gt_bboxes_xy - 1 / 2 * gt_bboxes_wh, gt_bboxes_xy + 1 / 2 * gt_bboxes_wh])
            anchor_bboxes_xy = anchor_bboxes[0:2]
            anchor_bboxes_wh = anchor_bboxes[2:4]
            anchor_bboxes = np.concatenate(
                [anchor_bboxes_xy - 1 / 2 * anchor_bboxes_wh, anchor_bboxes_xy + 1 / 2 * anchor_bboxes_wh])
            #print(anchor_bboxes)

            gt_image = CVShowBoxes(image, [gt_bboxes], waitkey=-1)
            anchor_image = CVShowBoxes(image, [anchor_bboxes], waitkey=-1)
            if class_id != 0:
                cv2.imwrite("Target_Anchor_images/"+str(i)+".jpg",anchor_image)
                cv2.imwrite("Target_GT_Images/" + str(i) + ".jpg", gt_image)
            cv2.imwrite("GT_images/"+str(i)+".jpg",gt_image)
            cv2.imwrite("ANCHOR_images/" + str(i) + ".jpg", anchor_image)
            # print(label[i,0:21])
            print(self.classes[class_id])


if __name__ == '__main__':

    images = np.load("npy/image.npy")
    labels = np.load("npy/label.npy")
    image = images[6, :, :, :]
    label = labels[6, :, :]
    ssdDecoder = SSDDecoder()
    ssdDecoder.decode(image, label)

    image_path_list = GetAllImagesPath("/home/jade/PycharmProject/Github/SSD-Tensorflow2.0/ANCHOR_images/")
    image_list = []
    for i in range(len(image_path_list)):
        if i > (38*38*4+ 19 * 19 * 6)and i < (38*38*4+ 19 * 19 * 6) + 100:
            image_list.append(os.path.join("/home/jade/PycharmProject/Github/SSD-Tensorflow2.0/ANCHOR_images/",str(i)+".jpg"))
    compose_gif(image_list,"gif/anchor_19.gif",fps=10)


