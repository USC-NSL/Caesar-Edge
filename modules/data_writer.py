import os 
import numpy as np
import sys 
from time import time 
from modules.data_reader import DataReader

''' Write metadata to file 

Input: 
     {'frame_id': frame_id, xxx}
    xxx means you can put whatever k-v entry

Output: 
    an npy file named in Setup
'''
class DataWriter:
    data = []

    def Setup(self, file_path='meta.npy'):
        if os.path.exists(file_path):
            self.log('WARNING: the file %s exist!' % file_path)
        
        self.fout = open(file_path, 'w')
        self.log('init')


    def PreProcess(self, input):
        if input and 'frame_id' in input:
            self.data.append(input)


    def save(self):
        self.log('saving data....')
        np.save(self.fout, self.data)
        self.data = []
        self.fout.close()
        self.log('ended')
        

    def log(self, s):
        print('[DataWriter] %s' % s)



''' UNIT TEST '''
if __name__ == '__main__':
    repeat = 100
    reader = DataReader()
    reader.Setup('test/video.mp4')
    data = []
    for i in xrange(repeat):
        data.append(reader.PostProcess()['meta'])

    writer = DataWriter()
    writer.Setup()
    cur_time = time()
    for d in data:
        writer.PreProcess(d)
    writer.save()
    print('FPS: %.1f' % (float(repeat) / float(time() - cur_time)))
    print('done')