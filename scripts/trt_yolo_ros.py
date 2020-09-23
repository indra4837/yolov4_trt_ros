#!/usr/bin/env python

import rospy
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import VisionInfo
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from yolo_object_detect.trt_yolo import TrtYOLOv4

import os


class yolov4_detection:
    def __init__(self):
        self.image_sub = rospy.Subscriber(
            Image, "/video_source/raw", 5, self.img_callback)
        self.detection_pub = rospy.Publisher(
            "detections", Detection2DArray, queue_size=25)
        self.overlay_pub = rospy.Publisher("overlay", Image, queue_size=2)
        self.info_pub = rospy.Publisher(
            "vision_info", VisionInfo, queue_size=1)
        self.read_params()
        self.bridge = CvBridge()
        self.model = TrtYOLOv4(self.model_file)

    def read_params(self):
        """ Read paramaters from config file
        """
        self.model_file = rospy.get_param("~model_file", "yolov4-416")

    def img_callback(self, ros_img):
        """ Callback that calls the TrtYOLOv4 model function """

        overlay_img = Image()

        # ros image to openCV image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img, desired_encoding='passthrough')
            rospy.logdebug("Image converted for processing")
        except CvBridgeError as e:
            rospy.logdebug("Failed to convert image %s", str(e))

        if cv_img is not None:
            rospy.logdebug("Processing frame")
            cv_img, boxes, confs, clss = self.model(cv_img)
            rospy.loginfo(
                'Boxes: {} \nConfidence: {} \nClass: {}'.format(boxes, confs, clss))

            # ros image to openCV image
            try:
                ros_img = self.bridge.cv2_to_imgmsg(cv_img, encoding="passthrough")
                self.overlay_pub.publish(overlay_img)
            except CvBridgeError as e:
                rospy.logdebug("Failed to convert image %s", str(e))


def main():
    yolov4_detection()
    rospy.init_node('yolov4_detection', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
