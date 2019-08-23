from nets.vgg import VGGBase
from tensorflow.python.keras import Model
import tensorflow as tf
from tensorflow.python.keras.layers import *
from utils.keras_layer_l2Normalization import L2Normalization
from utils.keras_layer_AnchorBoxes import AnchorBoxes
from utils.keras_layer_boxes import Anchors
from datasetopeation.jadeVocTFRecord import LoadVOCTFRecord
anchors = Anchors(args=None)
n_boxes = anchors.layer_boxes()


class SSD512(Model):
    def __init__(self,args):
        self.num_classes = args.num_classes + 1
        self.image_size = args.image_size
        super(SSD512, self).__init__()
        self.vgg_base = VGGBase()
        self.pool5 = MaxPool2D((3, 3), strides=(1, 1), padding="same", name="pool5")  # pool5层与vgg 有点区别
        # 在conv5_3特征图上做扩张卷积，此时扩张率为6的卷积，感受野相当于9*9，同时可以减少参数
        self.fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')
        self.fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', name="fc7")

        self.conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name="conv6_1")
        self.conv6_padding = ZeroPadding2D(padding=((1, 1), (1, 1)), name="conv6_padding")
        self.conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation="relu", padding="valid", name="conv6_2")
        # 注意填充方式有所区别

        self.conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="conv7_1")
        self.conv7_padding = ZeroPadding2D(padding=((1, 1), (1, 1)), name="conv7_padding")
        self.conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', name="conv7_2")

        self.conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')
        self.conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', name="conv8_2")

        self.conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv9_1')
        self.conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', name="conv9_2")

        # l2 normalization
        self.conv4_3_norm = L2Normalization(gamma_init=20, name="conv4_3_norm")

        self.block_mbox_conf = []
        self.block_mbox_loc = []
        self.block_mbox_prior_box = []

        self.block_mbox_conf_reshape = []
        self.block_mbox_loc_reshape = []
        self.block_mbox_prior_box_reshape = []

        self.block_mbox_name = ["conv4_3_norm_mbox", "fc7_mbox",
                                     "conv6_2_mbox", "conv7_2_mbox",
                                     "conv8_2_mbox", "conv9_2_mbox"]

        #conf 指的就是每一个框属于某一个类的conf
        for i in range(len(n_boxes)):

            #conf 的 shape (batch,height,width,n_boxes * num_classes)
            self.block_mbox_conf.append(Conv2D(n_boxes[i] * self.num_classes, (3, 3), padding="same",  name=self.block_mbox_name[i]+"_conf"))
            # 每个box里面有四个坐标值，所以坐标的预测值深度为 n_boxes * 4

            #loc 的 shape (batch,height,width,n_boxes*4)
            self.block_mbox_loc.append(Conv2D(n_boxes[i] * 4, (3,3),padding="same",name=self.block_mbox_name[i]+"_loc"))


            #anchor 的 shape (batch,height,width,n_boxes,8)
            self.block_mbox_prior_box.append(AnchorBoxes(self.image_size[0],self.image_size[1],
                                                         args.this_scale[i],args.this_scale[i+1],
                                                         aspect_ratios=args.aspect_ratios[i],
                                                         two_boxes_for_ar1=args.two_boxes_for_ar1,
                                                         this_steps=args.this_steps[i],this_offsets=args.this_offsets[i],
                                                         clip_boxes=args.clip_boxes,variances=args.variances,
                                                         coords=args.coords,normalize_coors=args.normalize_coors,name=self.block_mbox_name[i] + "_priorbox"
                                                         ))

            #Reshape conf reshape
            #shape = (batch,height*width*n_boxes,n_classes)
            self.block_mbox_conf_reshape.append(Reshape((-1,self.num_classes),name=self.block_mbox_name[i] + "_conf_reshape"))
            #Reshape loc reshape
            self.block_mbox_loc_reshape.append(Reshape((-1,4),name=self.block_mbox_name[i] + "_loc_reshape"))
            #Reshape prior_box
            self.block_mbox_prior_box_reshape.append(Reshape((-1,8),name=self.block_mbox_name[i] + "_prior_boxes_reshape"))

        self.mbox_conf = Concatenate(axis=1,name='mbox_conf')
        self.mbox_loc = Concatenate(axis=1,name="mbox_loc")
        self.mbox_priorbox = Concatenate(axis=1,name='mbox_priorbox')

        #对于类别预测，使用Softmax层
        self.mbox_conf_softmax = Activation('softmax',name='mbox_conf_softmax')

        #output_shape = (batch,n_boxes_total,n_classes + 4 + 8)

        self.predictions = Concatenate(axis=2,name="predictions")





    def call(self, x):
        end_points = []
        x, conv4_3 = self.vgg_base(x)
        end_points.append(conv4_3)
        x = self.pool5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        end_points.append(x)
        x = self.conv6_1(x)
        x = self.conv6_padding(x)
        x = self.conv6_2(x)
        end_points.append(x)
        x = self.conv7_1(x)
        x = self.conv7_padding(x)
        x = self.conv7_2(x)
        end_points.append(x)
        x = self.conv8_1(x)
        x = self.conv8_2(x)
        end_points.append(x)
        x = self.conv9_1(x)
        x = self.conv9_2(x)
        end_points.append(x)
        mbox_conf_reshapes = []
        mbox_loc_reshapes = []
        mbox_prior_reshapes = []
        for i in range(len(n_boxes)):
            mbox_conf = self.block_mbox_conf[i](end_points[i])
            mbox_loc = self.block_mbox_loc[i](end_points[i])
            mbox_prior = self.block_mbox_prior_box[i](mbox_loc)
            mbox_conf_reshape = self.block_mbox_conf_reshape[i](mbox_conf)
            mbox_loc_reshape = self.block_mbox_loc_reshape[i](mbox_loc)
            mbox_prior_reshape = self.block_mbox_prior_box_reshape[i](mbox_prior)
            mbox_conf_reshapes.append(mbox_conf_reshape)
            mbox_loc_reshapes.append(mbox_loc_reshape)
            mbox_prior_reshapes.append(mbox_prior_reshape)
        mbox_confs = self.mbox_conf(mbox_conf_reshapes)
        mbox_locs = self.mbox_loc(mbox_loc_reshapes)
        mbox_priors = self.mbox_priorbox(mbox_prior_reshapes)

        mbox_conf_softmax = self.mbox_conf_softmax(mbox_confs)
        predictions = self.predictions([mbox_conf_softmax,mbox_locs,mbox_priors])
        #输出值为(batch,8732,num_classes + 1 + 8 + 4)
        return predictions



if __name__ == '__main__':
    import argparse
    paraser = argparse.ArgumentParser(description="SSD")
    paraser.add_argument("--num_classes", default=10,help="num_classes")
    paraser.add_argument("--image_size", default=[300,300], help="image_size")
    paraser.add_argument("--this_scale", default= [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], help="this_scale")
    paraser.add_argument("--aspect_ratios", default=[[1.0,2.0,0.5], [1.0,2.0,0.5,3.0,1.0/3.0], [1.0,2.0,0.5,3.0,1.0/3.0],
                                        [1.0,2.0,0.5,3.0,1.0/3.0], [1.0,2.0,0.5],[1.0,2.0,0.5]], help="aspect_ratios")
    paraser.add_argument("--two_boxes_for_ar1", default=True, help="two_boxes_for_ar1")
    paraser.add_argument("--this_steps", default=[8,16,32,64,100,300], help="this_steps")
    paraser.add_argument("--this_offsets", default= [0.5,0.5,0.5,0.5,0.5,0.5], help="this_offsets")
    paraser.add_argument("--clip_boxes", default=False, help="clip_boxes")
    paraser.add_argument("--variances", default= [0.1,0.1,0.2,0.2], help="variances")
    paraser.add_argument("--coords", default="centroids", help="coords")
    paraser.add_argument("--normalize_coors", default=True, help="normalize_coors")
    args = paraser.parse_args()
    ssd512 = SSD512(args)
    input = tf.zeros(shape=[32,300,300,32],dtype=tf.float32)
    ssd512.train()
    x = ssd512.predict(input)


    print(x)
