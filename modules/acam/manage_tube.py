import os 
import cv2 
import numpy as np
import sys 

'''
A class that prepares the tubes frames for action detection
'''
class TubeManager:
    new_tubes = {}  # new tubes prepared for action detecion

    def __init__(self, act_update_freq, act_timesteps):
        self.act_update_freq = act_update_freq
        self.act_timesteps = act_timesteps
        self.log('init')


    def add_frame(self, input):
        ''' Take input from tracker, update the internal tube queues

        Input: {'img': img_np_array, 
                'meta':{
                        'frame_id': frame_id, 
                        'obj':[{
                                'box': [x0,y0,x1,y1],
                                'tid': track_id
                                }]
                        }
                }
        '''
        # TODO 


    def new_tubes(self):
        return self.new_tubes


    def clear_new_tubes(self):
        self.new_tubes = {}


    def log(self, s):
        print('[TubeManager] %s' % s)


''' UNIT TEST '''
if __name__ == '__main__':
    # TODO
    pass