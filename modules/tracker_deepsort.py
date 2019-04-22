import numpy as np 
from os.path import join 
import os 
import sys 
from time import time 
from modules.data_reader import DataReader
from modules.data_writer import DataWriter

DS_HOME = join(os.getcwd(), 'modules/deep_sort')
sys.path.insert(0, DS_HOME)
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import nn_matching 

'''
Input: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,w,h],
                        'conf': conf_score
                        'feature': feature_array
                        }]
                }
        }

Output: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,x1,y1],
                        'tid': track_id
                        }]
                }
        }
'''
class DeepSort:
    input = {}

    def Setup(self):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
        self.tracker = Tracker(metric, max_iou_distance=0.7, max_age=200, n_init=4)
        self.log('init')


    def PreProcess(self, input):
        self.input = input
        if not self.input:
            return 


    def Apply(self):
        ''' Extract features and update the tracker 
        ''' 
        if not self.input:
            return 

        detection_list = []
        for obj in self.input['meta']['obj']:
            detection_list.append(Detection(obj['box'], obj['conf'], obj['feature']))

        self.tracker.predict()
        self.tracker.update(detection_list)
        

    def PostProcess(self):
        output = self.input
        if not self.input:
            return ouptut

        output['meta']['obj'] = []
        for tk in self.tracker.tracks:
            if not tk.is_confirmed() or tk.time_since_update > 1:
                continue 
            left, top, width, height = map(int, tk.to_tlwh())
            track_id = tk.track_id
            output['meta']['obj'].append({'box':[left, top, width, height], 
                                            'tid':track_id})
        return output 


    def log(self, s):
        print('[DeepSort] %s' % s)


''' UNIT TEST '''
if __name__ == '__main__':
    fe = FeatureExtractor()
    fe.Setup()

    ds = DeepSort()
    ds.Setup()

    dr = DataReader()
    dr.Setup('test/video.mp4', 'obj_det_res.npy')

    dw = DataWriter()
    dw.Setup('track_res.npy')

    cur_time = time()
    cnt = 0 
    while True:
        d = dr.PostProcess()
        print(cnt)
        if not d:
            break 
        fe.PreProcess(d)
        fe.Apply()
        feature_objs = fe.PostProcess()
        ds.PreProcess(feature_objs)
        ds.Apply()
        res = ds.PostProcess()
        dw.PreProcess(res['meta'])
        cnt += 1

    print('FPS: %.1f' % (float(cnt) / float(time() - cur_time)))
    
    dw.save()
    print('done')
