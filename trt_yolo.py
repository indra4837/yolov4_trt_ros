#!/usr/bin/env python

import os
import time

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

import rospy
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import VisionInfo
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class yolov4:
    def __init__(self):
        self.image_sub = rospy.Subscriber(
            Image, "/video_source/raw", 5, self.img_callback)
        self.detection_pub = rospy.Publisher(
            "detections", Detection2DArray, queue_size=25)
        self.overlay_pub = rospy.Publisher("overlay", Image, queue_size=2)
        self.info_pub = rospy.Publisher(
            "vision_info", VisionInfo, queue_size=1)
        self.model_name = "yolov4-416"
        self.model_file = "yolo/yolov4-416.trt"
        self.bridge = CvBridge()
        self.category_num = 80
        self.conf_th = 0.5
        self.init_yolo()
   
    def img_callback(self, ros_img):
        """Continuously capture images from camera and do object detection """

        fps = 0.0
        tic = time.time()
            
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img, desired_encoding='bgr8')
            rospy.logdebug("Image converted for processing")
        except CvBridgeError as e:
            rospy.logdebug("Failed to convert image %s", str(e))
            
        if cv_img is not None:
            boxes, confs, clss = self.trt_yolo.detect(cv_img, self.conf_th)
            rospy.loginfo(boxes)
            cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
            cv_img = show_fps(cv_img, fps)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc

        # convert back to ros_img
        # publish to overlay topic

    def init_yolo(self):
        yolo_dim = self.model_name.split('-')[-1]
        if 'x' in yolo_dim:
            dim_split = yolo_dim.split('x')
            if len(dim_split) != 2:
                raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
            w, h = int(dim_split[0]), int(dim_split[1])
        else:
            h = w = int(yolo_dim)
        if h % 32 != 0 or w % 32 != 0:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

        self.trt_yolo = TrtYOLO(self.model_file, (h, w), self.category_num)
        cls_dict = get_cls_dict(self.category_num)
        self.vis = BBoxVisualization(cls_dict)


def main():
    yolov4()
    rospy.init_node('yolov4_detection', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()