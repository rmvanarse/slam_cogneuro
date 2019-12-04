# SLAM using object localization
Estimation of change in position using object tracking and a depth map.

### Description of Method
A Microsoft Kinect 360 is used. An RGB image is passed through YOLOv3 to detect 'objects of interest'. The locations of these objects are looked up in a depth map (obtained from freenect_stack with ROS Kinetic) to obtain the spherical coordinates of the object. If these objects are also found in consicutive images, the relative displacement in their vectors is used to calculate the displacement of the robot.


## Getting Started

### Prerequisites:

- opencv (cv2)
- darknet
- ROS Kinetic _(For getting data from Kinect)_
- freenect_stack
- YOLOv3.weights

### Cloning
Clone the repsitory on your local machine by running the following command

```git clone hhtp://github.com/rmvanarse/slam_cogneuro```

### Installations

**OpenCV 2.0** ```pip3 install opencv-python```
https://pypi.org/project/opencv-python

_(Requires: pip3)_

**freenect_stack**
http://wiki.ros.org/freenect_stack

**YOLOv3.weights**

Download the weights file from https://pjreddie.com/media/files/yolov3.weights

Alternatively, run the command ```wget https://pjreddie.com/media/files/yolov3.weights```

Save the yolov3.weights file in the YOLOv3 folder


## Running
Run the following commands after enterong the YOLOv3 folder by ```cd YOLOv3```
```
- python3 yolo_opencv.py --image1 <image1>.jpg --image2 <image2>.jpg --depthImage1 <depth1>.pgm --depthImage2 <depth2>.pgm --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt

- python3 yolo_opencv.py --image1 ../RGBD_img/Mul_exp1-1-rgb.png --image2 ../RGBD_img/Mul_exp1-2-rgb.png --depthImage1 ../RGBD_img/Mul_exp1-1-depth.png --depthImage2 ../RGBD_img/Mul_exp1-2-depth.png --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
```
Example:
```
python3 yolo_opencv.py --image1 r-1.ppm --image2 r-2.ppm --depthImage1 d-1.pgm --depthImage2 d-2.pgm --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt

```
### Built with
- Python3
- ROS Kinetic
- YOLOv3
- Pytorch
- CV2
- Freenect

## Limitations
* Revolution around a single object cannot be detected.
* Algortihm needs to be modified to differentiate between Yaw and X-displacement of the robot.
* Requires multiple objects in the frame for a good efficiency
* Will have to be combined with other odometry methods.

## Contributers

**Rishikesh Vanarse** [rmvanarse](https://github.com/rmvanarse)
**Akhil Tarikere** [Akter8](https://github.com/Akter8)
**Srisreyas Sundaresan** [SriSreyas](https://github.com/SriSreyas)
**Pranav Guruprasad**
