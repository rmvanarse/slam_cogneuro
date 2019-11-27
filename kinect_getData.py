"""

"""

import rospy
import time
import cv2

from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
from cv_bridge import CvBridge, CvBridgeError


#Global

num_images = 3
interval = 0.5
last_response = time.time()

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
		print("Success!")
		cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
		cv2.imwrite('KinectResults/Test.jpg', cv2_img)
		last_response = time.time()




rospy.init_node('Vision')

bridge = CvBridge()


sub_rgb = rospy.Subscriber('/camera/rgb/image_color', Image, callback_rgb)

rospy.spin()

