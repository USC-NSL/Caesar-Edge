from module.deep_sort.deep_sort import Detection
from module.deep_sort.deep_sort import Tracker
from module.deep_sort import nn_matching
import numpy as np 

# Download the model file to 'checkpoints/'
DEEPSORT_MODEL = 'checkpoints/mars-small128.pb'

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
        features = self.encoder(self.img, self.ds_boxes)

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
            left, top, width, height = tk.to_tlwh()
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
    import cv2 
    img = cv2.imread(sys.argv[1])
    h, w, _ = img.shape
    bbox = [w/4, h/4, w/2, h/2]
    data = {'img':img, 
            'meta':{'frame_id':0, 'obj':['box':bbox, 'label':'person', 'conf':0.9]}}
    for i in xrange(10):
        ds.PreProcess(data)
        ds.Apply()
        print(ds.PostProcess()['meta'])
