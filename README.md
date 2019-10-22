# SLAM using object localization

Estimation of change in position using object tracking and a depth map

##TODO

- [ ] Get class locations from YOLO
- [ ] Get depthmap from SharpNet/ Monodepth
- [ ] Algorithm for matching corresponding objects in different frames
- [ ] Get Kinect data
- [ ] Vector equations for position change

## Dependencies:

* opencv
* imageio?
* pytorch

## Running
YOLO:
```
python3 yolo_opencv.py --image <image>.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
```
