**THIS REPO IS DEPRECATED. PLEASE USE THE [UPDATED VERSION](https://github.com/USC-NSL/Caesar)**

# Caesar-For Edge Computing

## How to Use
- Clone this repo to your machine ```git clone --recurse-submodules https://github.com/USC-NSL/Caesar-Edge.git```
- Download the checkpoint models (see below)
- Double check the model path in these files to make sure they are same as yours: ```object_detector_yolo(#11-13)```, ```object_detector_ssd(#11)```, ```tracker_deepsort(#10)```, ```action_detection_acam(#12)```
- Rune ```pipeline.py```

## Checkpoint Preparation
All the NN model files should be put into the "checkpoints/" folder. You can download the models in these links:
- [SSD](https://drive.google.com/open?id=0B0qPCUZ-3YwWT1RCLVZNN3RTVEU)
- [YOLOv2_cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg), [YOLOv2_weights](https://pjreddie.com/media/files/yolov2.weights)
- [DeepSORT](https://drive.google.com/open?id=1m2ebLHB2JThZC8vWGDYEKGsevLssSkjo)
- [ACAM](https://drive.google.com/open?id=138gfVxWs_8LhHiVO03tKpmYBzIaTgD70)
- [ReID2](https://drive.google.com/file/d/1-2KHQms_RHYIrHwFQg-Vln-NEyhjj5_p/view?usp=sharing)

## Requirements
See ```requirements.txt``` for python packages. 
- For SSD, make sure your ```tensorflow-estimator``` is compatible with your local tf version. For example, ```tensorflow-estimator==1.10.12``` works for ```tensorflow-gpu=1.12.0```. Otherwise the tf will complain "tf.estimator package not installed"

## Components
One module's output will go to the next one
- Video Reader
- Object Detection ([SSD](https://github.com/balancap/SSD-Tensorflow), [YOLO](https://github.com/thtrieu/darkflow))
- Tracking ([DeepSORT](https://github.com/nwojke/deep_sort))
- Action Detection ([ACAM](https://github.com/oulutan/ACAM_Demo/blob/master/README.md))

## Performance
- SSD512: 25 FPS
- YOLOv2: 22 FPS
- DeepSORT: 49 FPS
- ACAM: 60 FPS/Tube