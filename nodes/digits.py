#!/usr/bin/env python

import torch
from torch import nn, tensor
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
from std_msgs.msg import Int64

bridge = CvBridge()

model = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 100, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(100, 100, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(4),
        nn.Flatten(),
        nn.Linear(100,80),
        nn.Linear(80,60),
        nn.Linear(60,40),
        nn.Linear(40,20),
        nn.Linear(20,10))


def image_callback(msg):
	#print("Received an image!")
	pub = rospy.Publisher('prediction', Int64, queue_size=10)
	try:
		# Convert your ROS Image message to OpenCV2
		img = bridge.imgmsg_to_cv2(msg, "bgr8")
	except CvBridgeError:
		print("ERROR!!")
	grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.threshold(grayed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	#img = cv2.bitwise_not(img)
	img = cv2.resize(img, (28,28), interpolation= cv2.INTER_NEAREST)/255
	img = np.expand_dims(img, 0)
	img = np.expand_dims(img, 0)
	data = torch.tensor(img)
	data = data.float()
	global model
	pred = model(data)
	pb = torch.exp(pred)
	probab = list(pb.detach().numpy()[0])
	pub.publish(probab.index(max(probab)))
	
	

def model_func():
	path = rospy.get_param('~path')      
	image_topic = rospy.get_param('~image_topic')
	global model
	model.load_state_dict(torch.load(path))
	# Set up your subscriber
	rospy.Subscriber(image_topic, Image, image_callback)
	rospy.spin()
	
	 


if __name__ == '__main__':
	rospy.init_node('image', anonymous=True)
	try:
		model_func()
	except rospy.ROSInterruptException:
		pass
