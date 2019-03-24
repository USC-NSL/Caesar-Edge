import numpy as np 
from os.path import join 
import os 
import sys 
from time import time 
from modules.data_reader import DataReader
from modules.data_writer import DataWriter

# Download the model file to 'checkpoints/'
DEEPSORT_MODEL = join(os.getcwd(),'checkpoints/deepsort/mars-small128.pb')

DS_HOME = join(os.getcwd(), 'modules/deep_sort')
sys.path.insert(0, DS_HOME)
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import nn_matching 
# The original DS tools folder doesn't have init file, add it
fout = open(join(DS_HOME, 'tools/__init__.py'), 'w')
fout.close()
from tools.generate_detections import create_box_encoder

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
                        'box': [x0,y0,x1,y1],
                        'tid': track_id
                        }]
                }
        }
'''
class DeepSort:
    ds_boxes = []
    scores = []
    input = {}

    def Setup(self):
        self.encoder = create_box_encoder(DEEPSORT_MODEL, batch_size=16)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
        self.tracker = Tracker(metric, max_iou_distance=0.7, max_age=200, n_init=4)
        self.log('init')


    def PreProcess(self, input):
        self.input = input
        if not self.input:
            return 

        boxes = input['meta']['obj']
        self.ds_boxes = [[b['box'][0], b['box'][1], 
                        b['box'][2]-b['box'][0], b['box'][3]-b['box'][1]] for b in boxes]
        self.scores = [b['conf'] for b in boxes]
        

    def Apply(self):
        ''' Extract features and update the tracker 
        ''' 
        if not self.input:
            return 
        features = self.encoder(self.input['img'], self.ds_boxes)

        detection_list = [Detection(self.ds_boxes[i], self.scores[i], features[i]) 
                                                    for i in xrange(len(self.ds_boxes))]
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
        ds.PreProcess(d)
        ds.Apply()
        objs = ds.PostProcess()
        dw.PreProcess(objs['meta'])
        cnt += 1

    print('FPS: %.1f' % (float(cnt) / float(time() - cur_time)))
    
    dw.save()
    print('done')
