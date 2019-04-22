import os 
import cv2 
import numpy as np
import sys 
import collections 

'''
A class that prepares the tubes frames for action detection
'''
class TManager:

    def __init__(self, cache_size=32, min_tube_len=16):
        self.cache_size = cache_size
        self.min_tube_len = min_tube_len  
        self.tubes = {}
        self.active_actors = set()
        self.frames = collections.deque()
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
        img, frame_id = input['img'], input['meta']['frame_id']
        self.frames.append(img)
        if len(self.frames) > self.min_tube_len:
            self.frames.popleft() 

        for tube in input['meta']['obj']:
            id, box = tube['tid'], tube['box']
            if not id in self.tubes:
                self.tubes[id] = collections.deque()
            self.tubes[id].append({'box':box, 'fid':frame_id})
            while self.tubes[id][0]['fid'] <= frame_id - self.cache_size:
                self.tubes[id].popleft()
            if len(self.tubes[id]) > self.min_tube_len:
                self.active_actors.add(id)
            elif id in self.active_actors:
                self.active_actors.remove(id)


    def get_norm_roi(self, mid_box):
        ''' Calculate normalized ROI, and the edge size 
        '''
        left, top, right, bottom = mid_box

        edge = max(bottom - top, right - left) / 2. * 1.5 # change this to change the size of the tube

        cur_center = (top+bottom)/2., (left+right)/2.
        context_top, context_bottom = cur_center[0] - edge, cur_center[0] + edge
        context_left, context_right = cur_center[1] - edge, cur_center[1] + edge

        normalized_top = (top - context_top) / (2*edge)
        normalized_bottom = (bottom - context_top) / (2*edge)

        normalized_left = (left - context_left) / (2*edge)
        normalized_right = (right - context_left) / (2*edge)

        norm_roi = [normalized_top, normalized_left, normalized_bottom, normalized_right]

        return edge, norm_roi


    def get_roi(self):
        ''' For each active actor, calcualte the region-of-interest for each frame 
        '''
        norm_roi = np.zeros([len(self.active_actors), 4])
        temporal_roi = np.zeros([len(self.active_actors), self.cache_size, 4])

        cnt = 0 
        for i in self.active_actors:
            tube = self.tubes[i]
            tube_len = len(tube)
            mid_box = tube[tube_len // 2]['box']
            edge, norm_roi[cnt] = self.get_norm_roi(mid_box)

            # get temporal ROI
            rois = []
            for j in xrange(self.cache_size):
                box = tube[max(0, j + tube_len - self.cache_size)]['box']
                left, top, right, bottom = box
                cur_center = (top+bottom)/2., (left+right)/2.
                top, bottom = cur_center[0] - edge, cur_center[0] + edge
                left, right = cur_center[1] - edge, cur_center[1] + edge
                rois.append([top, left, bottom, right])
            temporal_roi[cnt] = np.stack(rois, axis=0)
            cnt += 1

        return norm_roi, temporal_roi


    def new_tube_data(self):
        ''' Return the data of current new tubes 

        Output: tuple of four
            frames -        a list of whole frames (np array)
            temporal_roi -  nparry([actor_num, act_timesteps, 4]), 4 is (y0,x0,y1,x1)
            norm_roi -      nparray([actor_num, 4]), 4 is (y0,x0,y1,x1) 
            actor_boxes -   {'box':most_recent_box, 'tid':tid} of each act performer
        '''
        frames = np.expand_dims(np.stack(list(self.frames), axis=0), axis=0)
        norm_roi, temporal_roi = self.get_roi()
        actor_boxes = [{'box':[self.tubes[i][-1]['box']], 'tid':i} for i in self.active_actors]

        return frames, temporal_roi, norm_roi, actor_boxes


    def has_new_tube(self):
        return len(self.active_actors)


    def log(self, s):
        print('[TManager] %s' % s)
