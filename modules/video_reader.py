import os 
import cv2 
import numpy as np
import sys 

'''
Input: Video file name 
Output: {'img':img_np_array, 'meta':{'frame_id':frame_id}}
'''
class VideoReader:
    src = ''
    cap = None 
    frame_id = 0
    end_of_video = False 


    def Setup(self, video_path=''):
        if not video_path.isdigit() and not os.path.exists(video_path):
            self.log('Cannot load video!')
            self.end_of_video = True
            return 
        self.src = int(video_path) if self.src.isdigit() else video_path
        self.cap = cv2.VideoCapture(video_path)
        self.log('init')


    def PostProcess(self):
        if end_of_video:
            return {}

        ret, frame = self.cap.read()
        self.frame_id += 1
        if not ret:
            self.log('End of video')
            self.end_of_video = True 
            return {}
        return {'img': frame, 'meta': {'frame_id':self.frame_id}}


    def log(self, s):
        print('[VideoReader] %s' % s)



''' UNIT TEST '''
if __name__ == '__main__':
    reader = VideoReader(sys.argv[1])
    for i in xrange(100):
        print(reader.PostProcess()['meta'])