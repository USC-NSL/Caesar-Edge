# Darkflow should be installed from: https://github.com/thtrieu/darkflow
from darkflow.net.build import TFNet
import numpy as np

# Place your downloaded cfg and weights under "checkpoints/"
YOLO_CONFIG = 'checkpoints/yolo_cfg'
YOLO_MODEL = 'checkpoints/yolo_cfg/yolo.cfg'
YOLO_WEIGHTS = 'checkpoints/yolo.weights'
GPU_ID = 0
GPU_UTIL = 0.5
YOLO_THRES = 0.4
YOLO_PEOPLE_LABEL = 'person'

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
class YOLO:
    dets = []
    input = {}

    def Setup(self):
        opt = { "config": YOLO_CONFIG,  
                "model": YOLO_MODEL, 
                "load": YOLO_WEIGHTS, 
                "gpuName": GPU_ID,
                "gpu": GPU_UTIL,
                "threshold": YOLO_THRES
            }
        self.tfnet = TFNet(opt)
        self.log('init')


    def PreProcess(self, input):
        self.input = input 


    def Apply(self):
        if self.input:
            self.dets = self.tfnet.return_predict(self.input['img'])   
        

    def PostProcess(self):
        output = self.input
        if not self.input:
            return output 

        output['meta']['obj'] = []
        for d in self.dets:
            if d['label'] != YOLO_PEOPLE_LABEL:
                continue 
            self.output['meta']['obj'].append({'box':[int(d['topleft']['x']), int(d['topleft']['y']),
                                                    int(d['bottomright']['x']), int(d['bottomright']['y'])],
                                                'label': d['label'],
                                                'conf': d['confidence']})
        return output


    def log(self, s):
        print('[YOLO] %s' % s)


''' UNIT TEST '''
if __name__ == '__main__':
    yolo = YOLO()
    yolo.Setup()
    import cv2
    img = cv2.imread(sys.argv[1]) # read an image as the input 
    data = {'img':img, 'meta':1}
    yolo.PreProcess(data)
    yolo.Apply()
    print(yolo.PostProcess()['meta'])
