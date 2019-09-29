from modules.data_reader import DataReader
from modules.object_detector_ssd import SSD
# from modules.object_detector_yolo import YOLO
from modules.reid_extractor import FeatureExtractor
from modules.tracker_deepsort import DeepSort
from modules.tube_manager import TubeManager
from modules.action_detector_acam import ACAM
import sys 

# ============ Video Input Modules ============
reader = DataReader()
reader.Setup(sys.argv[1])

# ============ Object Detection Modules ============
ssd = SSD()
ssd.Setup()

# yolo = YOLO()
# yolo.Setup()

object_detector = ssd

# ============ Tracking Modules ============
feature_extractor = FeatureExtractor()
feature_extractor.Setup()

deepsort = DeepSort()
deepsort.Setup()

tracker = deepsort

# ============ Action Detection Modules ============
tube_manager = TubeManager()
tube_manager.Setup()

acam = ACAM()
acam.Setup()

action_detector = acam


while(True):
    # Read input
    frame_data = reader.PostProcess()
    if not frame_data:  # end of video 
        break 

    # Obj detection module
    object_detector.PreProcess(frame_data)
    object_detector.Apply()
    obj_det_data = object_detector.PostProcess()

    # Tracking module
    feature_extractor.PreProcess(obj_det_data)
    feature_extractor.Apply()
    feature_data = feature_extractor.PostProcess()

    tracker.PreProcess(feature_data)
    tracker.Apply()
    track_data = tracker.PostProcess()

    # Action detection module 
    tube_manager.PreProcess(track_data)
    tube_manager.Apply()
    tube_data = tube_manager.PostProcess()

    action_detector.PreProcess(tube_data)
    action_detector.Apply()
    action_data = action_detector.PostProcess()

    if action_data:
        print(action_data['meta']['obj'])
