# SLAM using object localization

Estimation of change in position using object tracking and a depth map

##TODO

- [X] Get class locations from YOLO
- [ ] Get depthmap from Kinect
- [X] Algorithm for matching corresponding objects in different frames
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

## Limitations

* Revolution around a single object cannot be detected.
* Motion in Z-axis will require multiple objects on both sides of the center to cancel out the X and Y axes errors.
* Will have to be combined with other odometry methods.
