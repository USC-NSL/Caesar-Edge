import os 
import cv2 
import numpy as np
import sys 
from time import time 

''' Read video and metadata file (opt) and output the data sequence

Input: 
    - Video file path
    - Optional: metadata file path (an npy file)
            format: [{'frame_id': frame_id, xxx}]
                xxx means you can put whatever k-v entry
Output: 
    Default: {'img':img_np_array, 'meta':{'frame_id':frame_id}}

'''
class DataReader:
    src = ''
    cap = None
    data = [] 
    frame_id = 0
    end_of_video = False 


    def Setup(self, video_path='', file_path=''):
        if not video_path.isdigit() and not os.path.exists(video_path):
            self.log('Cannot load video!')
            self.end_of_video = True
            return 
        
        self.src = int(video_path) if self.src.isdigit() else video_path
        self.cap = cv2.VideoCapture(video_path)

        self.data_ptr = 0 
        if file_path and os.path.exists(file_path):
            self.data = np.load(open(file_path, 'r'))
        
        self.log('init')


    def PostProcess(self):
        if self.end_of_video:
            return {}

        ret, frame = self.cap.read()

        if not ret:
            self.log('End of video')
            self.end_of_video = True 
            return {}

        output = {'img': frame, 'meta': {'frame_id':self.frame_id}}
        while len(self.data) and self.data_ptr < len(self.data) and \
                self.data[self.data_ptr]['frame_id'] < self.frame_id:
            self.data_ptr += 1

        if len(self.data) and self.data_ptr < len(self.data) and \
                self.data[self.data_ptr]['frame_id'] == self.frame_id:
            output['meta'] = self.data[self.data_ptr]

        self.frame_id += 1
        return output 


    def log(self, s):
        print('[DataReader] %s' % s)



''' UNIT TEST '''
if __name__ == '__main__':
    repeat = 100
    reader = DataReader()
    file_path = sys.argv[1] if len(sys.argv) == 2 else ''
    reader.Setup('test/video.mp4', file_path)
    cur_time = time()
    for i in xrange(repeat):
        print(reader.PostProcess()['meta'])
    print('FPS: %.1f' % (float(repeat) / float(time() - cur_time)))