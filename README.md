# SLAM using object localization
Estimation of change in position using object tracking and a depth map

##TODO
- [X] Get class locations from YOLO
- [ ] Get depthmap from Kinect
- [X] Algorithm for matching corresponding objects in different frames
- [ ] Get Kinect data
- [X] Vector equations for position change
- [ ] Scale factor for the vector subtraction
- [ ] Scale the depth from the dataset/kinect to a comprehensible value

## Dependencies:
* opencv
* imageio?
* pytorch

## Running
YOLO (with two images as arguments):
```
python3 yolo_opencv.py --image1 <image1>.jpg --image2 <image2>.jpg --depthImage1 <depth1>.pgm --depthImage2 <depth2>.pgm --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
```
Example:
```
python3 yolo_opencv.py --image1 r-1.ppm --image2 r-2.ppm --depthImage1 d-1.pgm --depthImage2 d-2.pgm --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
```

## Limitations
* Revolution around a single object cannot be detected.
* Motion in Z-axis will require multiple objects on both sides of the center to cancel out the X and Y axes errors.
* Will have to be combined with other odometry methods.
