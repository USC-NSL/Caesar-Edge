import modules.acam.action_detector as act
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

CACHE_SIZE = 32      # number of consecutive frames 
MIN_TUBE_LEN = 16    # output a list of tube images every MIN_TUBE_LEN new frames

PRINT_TOP_K = 5      # show the top 5 possible action for current second 

'''
Input: {'img': img_np_array, 
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

Output: {'img': None, 
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

    def Setup(self):
        self.act_detector = act.Action_Detector('soft_attn')
        self.updated_frames, self.temporal_rois, self.temporal_roi_batch_indices, cropped_frames = \
                    self.act_detector.crop_tubes_in_tf_with_memory([CACHE_SIZE,
                                                                    VIDEO_HEI,VIDEO_WID,3], 
                                                                    CACHE_SIZE - MIN_TUBE_LEN)

        self.rois, self.roi_batch_indices, self.pred_probs = \
                    self.act_detector.define_inference_with_placeholders_noinput(cropped_frames)

        self.act_detector.restore_model(ACAM_MODEL)


    def PreProcess(self, input):
        self.input = input 


    def Apply(self):
        if not self.input:
            return 

        obj = self.input['meta']['obj']
        tube_num = len(obj['actor_boxes'])
        feed_dict = {self.updated_frames:           obj['frames'], 
                    self.temporal_rois:             obj['temporal_rois'],
                    self.temporal_roi_batch_indices:np.zeros(tube_num),
                    self.rois:                      obj['norm_rois'], 
                    self.roi_batch_indices:         np.arange(tube_num)}
        run_dict = {'pred_probs': self.pred_probs}

        out_dict = self.act_detector.session.run(run_dict, feed_dict=feed_dict)
        self.probs = out_dict['pred_probs']

        
    def PostProcess(self):
        output = self.input 
        if not self.input or not len(self.probs):
            return {}

        output['meta']['obj'] = self.input['meta']['obj']['actor_boxes']
        for i in xrange(len(output['meta']['obj'])):
            act_probs = self.probs[i]
            order = np.argsort(act_probs)[::-1]
            cur_actor_id = output['meta']['obj'][i]['tid']
            cur_results = []
            for pp in range(PRINT_TOP_K):
                cur_results.append((act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
            output['meta']['obj'][i]['act'] = cur_results

        return output


    def log(self, s):
        print('[ACAM] %s' % s)


''' UNIT TEST '''
if __name__ == '__main__':    
    from modules.tube_manager import TubeManager
    tm = TubeManager()
    tm.Setup()

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
        tm.PreProcess(d)
        tm.Apply()
        tubes = tm.PostProcess()
        if tubes:
            acam.PreProcess(tubes)
            acam.Apply()
            res = acam.PostProcess()
            dw.PreProcess(res['meta'])
        cnt += 1

    print('FPS: %.1f' % (float(cnt) / float(time() - cur_time)))
    
    dw.save()
    print('done')