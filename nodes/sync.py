#!/usr/bin/env python

import rospy
import message_filters
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, CameraInfo


def gotimage(image, camerainfo):
    return
    

def sync():
    image_sub = Subscriber("/camera/image_raw", Image)
    camera_sub = Subscriber("/camera/camera_info", CameraInfo)
    ats = ApproximateTimeSynchronizer([image_sub, camera_sub], queue_size=5, slop=0.1)
    ats.registerCallback(gotimage)
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('sync', anonymous=True)
    try:
        sync()
    except rospy.ROSInterruptException:
        pass
