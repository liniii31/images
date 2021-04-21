#!/usr/bin/env python

import torch
from torch import nn, tensor
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2


bridge = CvBridge()



def create_model():
	model = nn.Sequential(
		nn.Conv2d(1, 6, 5, padding=2),
		nn.ReLU(),
		nn.AvgPool2d(2, stride=2),
		nn.Conv2d(6, 16, 5, padding=0),
		nn.ReLU(),
		nn.AvgPool2d(2, stride=2),
		nn.Flatten(),
		nn.Linear(400, 120),
		nn.ReLU(),
		nn.Linear(120, 84),
		nn.ReLU(),
		nn.Linear(84, 10)
	)
	return model

model = create_model()


def image_callback(msg):
	print("Received an image!")
	try:
		# Convert your ROS Image message to OpenCV2
		img = bridge.imgmsg_to_cv2(msg, "bgr8")
	except CvBridgeError:
		print("ERROR!!")
	image_pub = rospy.Publisher("/input_image",Image)
	grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.threshold(grayed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	#img = cv2.bitwise_not(img)
	img = cv2.resize(img, (28,28), interpolation= cv2.INTER_NEAREST)/255
	cv2.startWindowThread()
	print(cv2.imshow('image',img))
	cv2.waitKey(6)
	img = np.expand_dims(img, 0)
	img = np.expand_dims(img, 0)
	data = torch.tensor(img)
	data = data.float()
	image_pub.publish(bridge.cv2_to_imgmsg(img, encoding="passthrough"))
	global model
	pred = model(data)
	print("---------------------------------------------------------------------------------------------------------")
	pb = torch.exp(pred)
	probab = list(pb.detach().numpy()[0])
	print("Predicted Digit =", probab.index(max(probab)))
	print("---------------------------------------------------------------------------------------------------------")
	



def ip():
	global model
	model.load_state_dict(torch.load("/home/shalini/catkin_ws/src/images/nodes/mnsit_model.pt"))
	# Set up your subscriber
	image_topic = "/image_raw"
	rospy.Subscriber(image_topic, Image, image_callback)
	print("Waiting for image")
	rospy.spin()
	cv2.destroyAllWindows()  



if __name__ == '__main__':
	rospy.init_node('image', anonymous=True)
	try:
		ip()
	except rospy.ROSInterruptException:
		pass

	