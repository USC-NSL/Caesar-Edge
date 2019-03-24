import modules.acam.action_detector as act
from modules.acam.manage_tube import TubeManager
import sys 
import numpy as np 
from os.path import join 
import os 
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

    def Setup(self):
        self.act_detector = act.Action_Detector('soft_attn')
        self.action_freq = 16           # update the act result every 16 frames 
        memory_size = self.act_detector.timesteps - self.action_freq         # 32 - 16

        self.updated_frames, self.temporal_rois, self.temporal_roi_batch_indices, cropped_frames = \
                    self.act_detector.crop_tubes_in_tf_with_memory([self.act_detector.timesteps,
                                                                    VIDEO_HEI,VIDEO_WID,3], 
                                                                    memory_size)

        self.ois, self.roi_batch_indices, self.pred_probs = \
                    self.act_detector.define_inference_with_placeholders_noinput(cropped_frames)

        self.act_detector.restore_model(ACAM_MODEL)
        self.tube_manager = TubeManager(self.action_freq, self.act_detector.timesteps)
        self.print_top_k = 5        # show top 5 actions for each tube 


    def PreProcess(self, input):
        self.input = input 
        if input:
            self.tube_manager.add_frame(input)


    def Apply(self):
        if not self.input or not self.tube_manager.new_tubes():
            return 

        tubes = self.tube_manager.new_tubes()
        feed_dict = {self.updated_frames:           tubes['cur_input_sequence'], 
                    self.temporal_rois:             tubes['temporal_rois_np'],
                    self.temporal_roi_batch_indices:np.zeros(tubes['num_actors']),  # size of tracks
                    self.rois:                      tubes['rois_np'], 
                    self.roi_batch_indices:         np.arange(tubes['num_actors'])}
        run_dict = {'pred_probs': self.pred_probs}

        out_dict = act_detector.session.run(run_dict, feed_dict=feed_dict)
        self.probs = out_dict['pred_probs']

        self.tube_manager.clear_new_tubes()

        
    def PostProcess(self):
        output = self.input 
        if not self.input:
            return output

        for i in xrange(len(self.probs)):
            act_probs = self.probs[bb]
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