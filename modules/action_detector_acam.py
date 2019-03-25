import modules.acam.action_detector as act
from modules.acam.manage_tube import TubeManager
import sys 
import numpy as np 
from os.path import join 
import os 
from time import time 
from modules.data_reader import DataReader
from modules.data_writer import DataWriter

# Download the model file to 'checkpoints/'
ACAM_MODEL = join(os.getcwd(),'checkpoints/model_ckpt_soft_attn_pooled_cosine_drop_ava-130')

# These two numbers should be same as your video input 
VIDEO_WID = 1920    
VIDEO_HEI = 1080

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

Output: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,x1,y1],
                        'tid': track_id,
                        'act': [(act_label, act_prob)]
                        }]
                }
        }
'''
class ACAM:
    input = {}
    probs = []
    actor_boxes = []

    def Setup(self):
        self.act_detector = act.Action_Detector('soft_attn')
        self.action_freq = 16           # update the act result every 16 frames 
        memory_size = self.act_detector.timesteps - self.action_freq         # 32 - 16

        self.updated_frames, self.temporal_rois, self.temporal_roi_batch_indices, cropped_frames = \
                    self.act_detector.crop_tubes_in_tf_with_memory([self.act_detector.timesteps,
                                                                    VIDEO_HEI,VIDEO_WID,3], 
                                                                    memory_size)

        self.rois, self.roi_batch_indices, self.pred_probs = \
                    self.act_detector.define_inference_with_placeholders_noinput(cropped_frames)

        self.act_detector.restore_model(ACAM_MODEL)
        self.tube_manager = TubeManager(cache_size=self.act_detector.timesteps, min_tube_len=self.action_freq)
        self.print_top_k = 5        # show top 5 actions for each tube 


    def PreProcess(self, input):
        self.input = input 
        if input:
            self.tube_manager.add_frame(input)


    def Apply(self):
        if not self.input or self.input['meta']['frame_id'] % self.action_freq != 0 or \
                                self.input['meta']['frame_id'] < self.act_detector.timesteps:
            return 

        if not self.tube_manager.has_new_tube():
            return 

        frames, temporal_rois, norm_rois, self.actor_boxes = self.tube_manager.new_tube_data()
        feed_dict = {self.updated_frames:           frames, 
                    self.temporal_rois:             temporal_rois,
                    self.temporal_roi_batch_indices:np.zeros(len(self.actor_boxes)),
                    self.rois:                      norm_rois, 
                    self.roi_batch_indices:         np.arange(len(self.actor_boxes))}
        run_dict = {'pred_probs': self.pred_probs}

        out_dict = self.act_detector.session.run(run_dict, feed_dict=feed_dict)
        self.probs = out_dict['pred_probs']

        
    def PostProcess(self):
        output = self.input 
        if not self.input or not len(self.probs):
            return {}

        output['meta']['obj'] = self.actor_boxes
        for i in xrange(len(self.actor_boxes)):
            act_probs = self.probs[i]
            order = np.argsort(act_probs)[::-1]
            cur_actor_id = output['meta']['obj'][i]['tid']
            cur_results = []
            for pp in range(self.print_top_k):
                cur_results.append((act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
            output['meta']['obj'][i]['act'] = cur_results

        self.probs = []
        return output


    def log(self, s):
        print('[ACAM] %s' % s)


''' UNIT TEST '''
if __name__ == '__main__':
    acam = ACAM()
    acam.Setup()

    dr = DataReader()
    dr.Setup('test/video.mp4', 'track_res.npy')

    dw = DataWriter()
    dw.Setup('act_res.npy')

    cur_time = time()
    cnt = 0 
    while True:
        d = dr.PostProcess()
        print(cnt)
        if not d:
            break 
        acam.PreProcess(d)
        acam.Apply()
        objs = acam.PostProcess()
        if objs:
            dw.PreProcess(objs['meta'])
        cnt += 1

    print('FPS: %.1f' % (float(cnt) / float(time() - cur_time)))
    
    dw.save()
    print('done')

