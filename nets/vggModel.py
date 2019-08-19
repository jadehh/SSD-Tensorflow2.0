#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/14 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/14  下午3:45 modify by jade

from nets.vgg import VGGNet16
from jade import *
import argparse

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.84
class VGGModel():
    def __init__(self,args):
        self.model_path = args.model_path
        self.prototxt_path = args.prototxt_path
        self.categorities,self.classes = ReadProTxt(self.prototxt_path)
        self.num_classes = len(self.classes[1:])
        self.model = self.load_model()
    def load_model(self):
        new_model = VGGNet16(classes=self.num_classes)
        new_model.build(input_shape=(None, 224, 224, 3))
        new_model.load_weights(self.model_path)
        print("restore the model from ",self.model_path)
        return new_model
    def predict(self,img):
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img = cv2.resize(img,(224,224))
        img = img.astype("float32")
        predictions = self.model.predict(np.array([img]))
        class_id = np.argmax(predictions[0])
        # print(np.argmax(predictions[0]))
        return self.categorities[class_id+1]["name"]

if __name__ == '__main__':
    paraser = argparse.ArgumentParser(description="VOC Classify")
    # genearl
    paraser.add_argument("--model_path",
                         default="/home/jade/Models/sdfgoodsClassify/sdfgoods_vgg16net_2019-08-16.h5",
                         help="path to load model")
    paraser.add_argument("--prototxt_path", default="/home/jade/Data/sdfgoods/sdfgoods.prototxt", help="path to labels")
    args = paraser.parse_args()
    vggModel = VGGModel(args)
    image_path_list = GetAllImagesPath("/home/jade/Data/sdfgoods/sdfgoods10/bkl-bklst-pz-yw-500ml")
    for image_path in image_path_list:
        img = cv2.imread(image_path)
        classify_result = vggModel.predict(img)
        print(classify_result)

