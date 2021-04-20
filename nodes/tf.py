#!/usr/bin/env python

import rospy
import math
import tf2_ros
import geometry_msgs.msg




def publisher():
	br = tf2_ros.TransformBroadcaster()
	t = geometry_msgs.msg.TransformStamped()
	# periodic rate: 
	r = rospy.Rate(10)
	while not rospy.is_shutdown():
		t1 = geometry_msgs.msg.TransformStamped()
		t1.header.stamp = rospy.Time.now()
		t1.header.frame_id = "map"
		t1.child_frame_id = "base_link"
		t1.transform.translation.x = 0
		t1.transform.translation.y = 0
		t1.transform.rotation.x = 0
		t1.transform.rotation.y = 0
		t1.transform.rotation.z = 0
		t1.transform.rotation.w = 1
		br.sendTransform(t1)
		r.sleep()


if __name__ == '__main__':
	rospy.init_node('tf_publisher', anonymous=True)
	try:
		
		publisher()
	except rospy.ROSInterruptException:
		pass
