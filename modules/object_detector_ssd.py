import sys
import numpy as np
import tensorflow as tf
from time import time 
from modules.data_reader import DataReader
from modules.data_writer import DataWriter
from os.path import join 
import os 

# Place your downloaded ckpt under "checkpoints/"
SSD_CKPT = join(os.getcwd(), 'checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt')

SSD_HOME = join(os.getcwd(), 'modules/SSD-Tensorflow') 
sys.path.insert(0, SSD_HOME)
from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

SSD_THRES = 0.4
SSD_NMS = 0.45
SSD_NET_SHAPE = (512, 512)
SSD_PEOPLE_LABEL = 15

'''
Input: {'img':img_np_array, 'meta':{'frame_id':frame_id}}

Output: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,x1,y1],
                        'label': label,
                        'conf': conf_score
                        }]
                }
        }
'''
class SSD:
    rclasses = []
    rbboxes = []
    rscores = []
    input = {}

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
                                        self.img_input, None, None, self.net_shape, data_format, 
                                        resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        
        self.image_4d = tf.expand_dims(image_pre, 0)
        self.bbx = bbox_img

        reuse = True if 'ssd_net' in locals() else None
        ssd_net = ssd_vgg_512.SSDNet()
        slim = tf.contrib.slim
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
        self.input = input 


    def Apply(self):
        if not self.input:
            return 

        rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d, self.pred, self.loc, self.bbx],
                                                                feed_dict={self.img_input: self.input['img']})

        self.rclasses, self.rscores, self.rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=self.thres, img_shape=self.net_shape, num_classes=self.total_classes, decode=True)
    
        self.rbboxes = np_methods.bboxes_clip(rbbox_img, self.rbboxes)
        self.rclasses, self.rscores, self.rbboxes = np_methods.bboxes_sort(self.rclasses, self.rscores, 
                                                                        self.rbboxes, top_k=400)
        self.rclasses, self.rscores, self.rbboxes = np_methods.bboxes_nms(self.rclasses, self.rscores, 
                                                                        self.rbboxes, nms_threshold=self.nms_thres)
        self.rbboxes = np_methods.bboxes_resize(rbbox_img, self.rbboxes)


    def PostProcess(self):
        output = self.input
        if not self.input:
            return output

        output['meta']['obj'] = []
        shape = self.input['img'].shape
        for i in xrange(self.rbboxes.shape[0]):
            if self.rclasses[i] != SSD_PEOPLE_LABEL:
                continue 
            bbox = self.rbboxes[i]
            p1 = (int(bbox[0] * shape[1]), int(bbox[1] * shape[0]))
            p2 = (int(bbox[2] * shape[1]), int(bbox[3] * shape[0]))
            output['meta']['obj'].append({'box':[p1[0],p1[1],p2[0],p2[1]], 'conf': self.rscores[i], 
                                                                            'label':self.rclasses[i]})
        return output 


    def log(self, s):
        print('[SSD] %s' % s)


''' UNIT TEST '''
if __name__ == '__main__':
    ssd = SSD()
    ssd.Setup()

    dr = DataReader()
    dr.Setup('test/video.mp4')

    dw = DataWriter()
    dw.Setup('obj_det_res.npy')

    cur_time = time()
    cnt = 0 
    while True:
        d = dr.PostProcess()
        print(cnt)
        if not d:
            break 
        ssd.PreProcess(d)
        ssd.Apply()
        objs = ssd.PostProcess()
        dw.PreProcess(objs['meta'])
        cnt += 1

    print('FPS: %.1f' % (float(cnt) / float(time() - cur_time)))
    
    dw.save()
    print('done')
