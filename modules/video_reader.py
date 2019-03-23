import os 
import cv2 
import numpy as np

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
            return {'img': self.empty_frame, 'meta': self.frame_id}

        ret, frame = self.cap.read()
        self.frame_id += 1
        if not ret:
            self.log('end of video')
            self.end_of_video = True 
            return {'img': self.empty_frame, 'meta': self.frame_id}
        return {'img': frame, 'meta': self.frame_id}


    def log(self, s):
        print('[VideoReader] %s' % s)


    def empty_frame(self):
        ''' return an empty frame
        '''
        return np.array([])