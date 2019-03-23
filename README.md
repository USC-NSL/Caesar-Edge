# Caesar-For Edge Computing

## Components
One module's output will go to the next one
- Video Reader
- Object Detection ([SSD](https://github.com/balancap/SSD-Tensorflow), [YOLO](https://github.com/thtrieu/darkflow))
- Tracking ([DeepSORT](https://github.com/nwojke/deep_sort))
- Action Detection ([ACAM](https://github.com/oulutan/ACAM_Demo/blob/master/README.md))

## Checkpoint Preparation
All the NN model files should be put into the "checkpoints/" folder. You can download the models in these links:
- [SSD](https://drive.google.com/open?id=0B0qPCUZ-3YwWT1RCLVZNN3RTVEU)
- [YOLOv2_cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg), [YOLOv2_weights](https://pjreddie.com/media/files/yolov2.weights)
- [DeepSORT](https://drive.google.com/open?id=1m2ebLHB2JThZC8vWGDYEKGsevLssSkjo)
- [ACAM](https://drive.google.com/open?id=138gfVxWs_8LhHiVO03tKpmYBzIaTgD70)