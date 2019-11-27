"""

"""

import rospy
import time
import cv2

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import UInt8
from cv_bridge import CvBridge, CvBridgeError


#Global

num_images = 3
interval = 0.5

#last_response is for RGB and last_response1 is for depth
last_response = time.time()
last_response1 = time.time()

RGB_Topic = '/camera/rgb/image_color'
Depth_Topic = '/camera/depth_registered/points'

def callback_rgb(msg):
	"""
	Arguments:
		msg: The message from the topic

	Returns: None

	Saves the latest published RGB image after every interval
	

	"""
	#print("Callback successful!")
	global last_response
	if(time.time() - last_response > interval):
		print("Success1!")
		cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
		cv2.imwrite('KinectResults/RGB.jpg', cv2_img)
		last_response = time.time()

def callback_depth(msg):
	"""
	Arguments:
		msg: The message from the topic

	Returns: None

	Saves the latest published RGB image after every interval
	

	"""
	#print("Callback successful!")
	global last_response1
	if(time.time() - last_response1 > interval):
		print("Success2!")

		#CORRECT THIS PART
		
		cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
		cv2.imwrite('KinectResults/Depth.jpg', cv2_img)
		last_response1 = time.time()




rospy.init_node('Vision')

bridge = CvBridge()


sub_rgb = rospy.Subscriber(RGB_Topic, Image, callback_rgb)

#Import required msg type and change from Image
sub_depth = rospy.Subscriber(Depth_Topic, Image, callback_depth)

rospy.spin()

