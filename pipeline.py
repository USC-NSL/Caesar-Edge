from modules.video_reader import VideoReader
from modules.object_detector_ssd import SSD
from modules.object_detector_yolo import YOLO
from modules.tracker_deepsort import DeepSort
from modules.action_detector_acam import ACAM
import sys 

# ============ Video Input Modules ============
video_reader = VideoReader()
video_reader.Setup(sys.argv[1])

# ============ Object Detection Modules ============
ssd = SSD()
ssd.Setup()

yolo = YOLO()
yolo.Setup()

object_detector = ssd

# ============ Tracking Modules ============
deepsort = DeepSort()
deepsort.Setup()

tracker = deepsort

# ============ Action Detection Modules ============
acam = ACAM()
acam.Setup()

action_detector = acam


while(True):
    # Read input
    frame_data = video_reader.PostProcess()

    # Obj detection module
    object_detector.PreProcess(frame_data)
    object_detector.Apply()
    obj_det_data = object_detector.PostProcess()

    # Tracking module
    tracker.PreProcess(obj_det_data)
    tracker.Apply()
    track_data = tracker.PostProcess()

    # Action detection module 
    action_detector.PreProcess(track_data)
    action_detector.Apply()
    action_data = action_detector.PostProcess()

    print(action_data['meta'])