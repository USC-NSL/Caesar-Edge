import os
import math
import random
import sys
import numpy as np
import tensorflow as tf
import cv2
from time import time 

slim = tf.contrib.slim

# The following line is specific to the SSD install path on your machine
# clone the repo from; https://github.com/balancap/SSD-Tensorflow
SSD_HOME = '/home/smc/SSD-Tensorflow'
sys.path.insert(0, SSD_HOME)
from ssd_nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

# Place your downloaded ckpt under "checkpoints/"
SSD_CKPT = 'checkpoints/SSD_512x512_ft_iter_120000.ckpt/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt'

SSD_THRES = 0.3
SSD_NMS = 0.45
SSD_NET_SHAPE = (512, 512)


class SSD:
    def Setup(self):
        self.ckpt = SSD_CKPT
        self.thres = SSD_THRES  
        self.nms_thres = SSD_NMS
        self.net_shape = SSD_NET_SHAPE

        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        
        self.isess = tf.InteractiveSession(config=config)

        data_format = 'NHWC'
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

        image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, self.net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        
        self.image_4d = tf.expand_dims(image_pre, 0)
        self.bbx = bbox_img

        reuse = True if 'ssd_net' in locals() else None
        ssd_net = ssd_vgg_512.SSDNet()
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            predictions, localisations, _, _ = ssd_net.net(self.image_4d, is_training=False, reuse=reuse)

        self.isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.isess, self.ckpt)

        self.pred = predictions
        self.loc = localisations
        self.ssd_anchors = ssd_net.anchors(self.net_shape)
        self.total_classes = 21

        self.log('init done ')


    def PreProcess(self, input):
        pass

    def Apply(self):
        rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d, self.pred, self.loc, self.bbx],
                                                                feed_dict={self.img_input: img})

        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=self.thres, img_shape=self.net_shape, num_classes=self.total_classes, decode=True)
    
        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=self.nms_thres)
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes
        

    def PostProcess(self):
        pass


    def log(self, s):
        print('[SSD] %s')


if __name__ == '__main__':
    args = {'ssd_home': '/home/smc/SSD-Tensorflow',
            'ssd_ckpt': '/home/smc/models/SSD_512x512_ft_iter_120000.ckpt/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
            'ssd_thres': 0.2,
            'ssd_nms': 0.45,
            'ssd_net_shape': (300,300),
            }

    ssd = SSD(args)

    img = cv2.imread(sys.argv[1])
    cur_time = time()
    for i in range(50):
        rclasses, rscores, rbboxes =  ssd.detect_frame(img)
    print('total time %f' % ((time() - cur_time) / 50.))
    
    visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
