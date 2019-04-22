from modules.acam.manage_tube import TManager
import sys 
import numpy as np 
from os.path import join 
import os 
from time import time 

# These two numbers should be same as your video input 
VIDEO_WID = 1920    
VIDEO_HEI = 1080

CACHE_SIZE = 32      # number of consecutive frames 
MIN_TUBE_LEN = 16    # output a list of tube images every MIN_TUBE_LEN new frames

'''
Input: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,x1,y1],
                        'tid': track_id
                        }]
                }
        }

Output: {'img': None, 
        'meta':{
                'frame_id': frame_id, 
                'obj':{
                        'frames': list of cv2 frames, 
                        'temporal_rois': list of boxes, 
                        'norm_rois': list of boxes,
                        'tube_boxes': list of boxes
                        }
                }
        }
'''

class TubeManager:
    input = {}
    output_obj = {}


    def Setup(self):
        self.tmanager = TManager(cache_size=CACHE_SIZE, min_tube_len=MIN_TUBE_LEN)


    def PreProcess(self, input):
        self.input = input 
        if input:
            self.tmanager.add_frame(input)


    def Apply(self):
        if not self.input or self.input['meta']['frame_id'] % MIN_TUBE_LEN != 0 or \
                                self.input['meta']['frame_id'] < CACHE_SIZE:
            return 

        if not self.tmanager.has_new_tube():
            return 

        frames, temporal_rois, norm_rois, actor_boxes = self.tmanager.new_tube_data()
        self.output_obj['frames'] = frames
        self.output_obj['temporal_rois'] = temporal_rois
        self.output_obj['norm_rois'] = norm_rois
        self.output_obj['actor_boxes'] = actor_boxes
        
        
    def PostProcess(self):
        output = self.input 
        if not self.input or not self.output_obj:
            return {}

        output['img'] = None
        output['meta']['obj'] = self.output_obj
        return output


    def log(self, s):
        print('[TM] %s' % s)