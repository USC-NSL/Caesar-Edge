import cv2 
import numpy as np  
import sys 
import tensorflow as tf  
from time import time 

import reid2.resnet_v1_50 as model
import reid2.fc1024 as head

'''
Input: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,x1,y1],
                        'label': label,
                        'conf': conf_score
                        }]
                }
        }

Output: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,w,h],
                        'conf': conf_score
                        'feature': feature_array
                        }]
                }
        }
'''


MODEL_PATH = 'checkpoints/model/checkpoint-25000'

class FeatureExtractor2:
    batch_size = 16
    img_crop_dimension = (128, 256)
    empty_img_crop = np.zeros([256, 128, 3])
    img_crops = []
    input = {}
    features = []


    def Setup(self):
        tf.Graph().as_default()
        self.sess = tf.Session()
        self.images = tf.zeros([self.batch_size, 256, 128, 3], dtype=tf.float32)
        self.endpoints, body_prefix = model.endpoints(self.images, is_training=False)
        with tf.name_scope('head'):
            self.endpoints = head.head(self.endpoints, 128, is_training=False)
        tf.train.Saver().restore(self.sess, MODEL_PATH)

        self.log('init')

    def PreProcess(self, input):
        self.input = input
        if not self.input:
            return 

        meta = input['meta']['obj']
        img = input['img']
        tmp_crops = []
        for m in meta:
            b = m['box']
            crop = cv2.resize(img[b[1]:b[3], b[0]:b[2], ::-1], self.img_crop_dimension)
            self.log(crop.shape)
            tmp_crops.append(crop)
            if len(tmp_crops) == self.batch_size:
                self.img_crops.append(np.array(tmp_crops))
                tmp_crops = []

        if tmp_crops:
            self.img_crops.append(
                np.array(
                    tmp_crops + 
                    [self.empty_img_crop for _ in range(self.batch_size - len(tmp_crops))]
                )
            )

    def Apply(self):
        if not self.input:
            return 
        for crops in self.img_crops:
            self.features.append(
                self.sess.run(
                    self.endpoints['emb'], 
                    feed_dict={self.images: crops}
                )
            )

    def PostProcess(self):
        output = self.input
        if not self.input:
            return ouptut

        for i in range(len(output['meta']['obj'])):
            b = output['meta']['obj'][i]['box']
            output['meta']['obj'][i]['box'] = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            output['meta']['obj'][i]['feature'] = self.features[i // self.batch_size][i % self.batch_size]
            
        self.img_crops = []
        self.features = []
        return output 

    def log(self, s):
        print('[FExtractor2] %s' % str(s))


if __name__ == '__main__':
    # test
    fe = FeatureExtractor2()
    fe.Setup()

    im = cv2.imread(sys.argv[1])
    h, w, _ = im.shape
    box = [w//5, h//5, w//5*4, h//5*4]
    
    for i in range(10):
        cur = time()    
        request = {
            'img': im,
            'meta': {
                'frame_id': 0,
                'obj': [
                    {
                        'box': box,
                        'label': 1,
                        'conf_score': 1.,
                    }
                ]
            }
        }
        fe.PreProcess(request)
        fe.Apply()
        _ = fe.PostProcess()
        print('TIME: %0.2f' % (time() - cur))