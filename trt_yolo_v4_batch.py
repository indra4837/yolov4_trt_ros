#!/usr/bin/env python

import os
import time

import cv2
import pycuda.autoinit  # For initializing CUDA driver
import pycuda.driver as cuda

from utils.yolo_classes import get_cls_dict
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins_batch import TrtYOLO

import rospy
import rospkg
from yolov4_trt_ros.msg import Detector2DArray
from yolov4_trt_ros.msg import Detector2D
from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class yolov4(object):
    def __init__(self):
        """ Constructor """

        self.bridge = CvBridge()
        self.init_params()
        self.init_yolo()
        self.cuda_ctx = cuda.Device(0).make_context()
        self.trt_yolo = TrtYOLO(
            (self.model_path + self.model), (self.h, self.w), self.category_num)

    def __del__(self):
        """ Destructor """

        self.cuda_ctx.pop()
        del self.trt_yolo
        del self.cuda_ctx

    def clean_up(self):
        """ Backup destructor: Release cuda memory """

        if self.trt_yolo is not None:
            self.cuda_ctx.pop()
            del self.trt_yolo
            del self.cuda_ctx

    def init_params(self):
        """ Initializes ros parameters """

        rospack = rospkg.RosPack()
        package_path = rospack.get_path("yolov4_trt_ros")
        #self.video_topic1 = rospy.get_param(
        #    "/video_topic1", "/zed/zed_node/left_raw/image_raw_color")
        self.video_topic1 = rospy.get_param(
            "/video_topic1", "video_source/raw")
        self.video_topic2 = rospy.get_param(
            "/video_topic2", "/zed/zed_node/left_raw/image_raw_color")
        self.video_topic3 = rospy.get_param(
            "/video_topic3", "/zed/zed_node/right_raw/image_raw_color")
        self.video_topic4 = rospy.get_param(
            "/video_topic4", "/zed/zed_node/right_raw/image_raw_color")
        self.model = rospy.get_param("/model", "yolov4Custom")
        self.model_path = rospy.get_param(
            "/model_path", package_path + "/yolo/")
        self.category_num = rospy.get_param("/category_number", 10)
        self.input_shape = rospy.get_param("/input_shape", "416")
        self.conf_th = rospy.get_param("/confidence_threshold", 0.5)
        self.show_img = rospy.get_param("/show_image", True)
        self.image_sub1 = rospy.Subscriber(
            self.video_topic1, Image, self.img_callback1, queue_size=1, buff_size=2**26)
        self.image_sub2 = rospy.Subscriber(
            self.video_topic2, Image, self.img_callback2, queue_size=1, buff_size=2**26)
        self.image_sub3 = rospy.Subscriber(
            self.video_topic3, Image, self.img_callback3, queue_size=1, buff_size=2**26)
        self.image_sub4 = rospy.Subscriber(
            self.video_topic4, Image, self.img_callback4, queue_size=1, buff_size=2**26)
        self.detection_pub = rospy.Publisher(
            "detections", Detector2DArray, queue_size=1)
        self.overlay_pub = rospy.Publisher(
            "/result/overlay", Image, queue_size=1)

    def init_yolo(self):
        """ Initialises yolo parameters required for trt engine """

        if self.model.find('-') == -1:
            self.model = self.model + "-" + self.input_shape

        yolo_dim = self.model.split('-')[-1]

        if 'x' in yolo_dim:
            dim_split = yolo_dim.split('x')
            if len(dim_split) != 2:
                raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
            self.w, self.h = int(dim_split[0]), int(dim_split[1])
        else:
            self.h = self.w = int(yolo_dim)
        if self.h % 32 != 0 or self.w % 32 != 0:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

        cls_dict = get_cls_dict(self.category_num)
        self.vis = BBoxVisualization(cls_dict)

    def img_callback1(self, ros_img):
        """Continuously capture images from camera and do object detection """

        # converts from ros_img to cv_img for processing
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img, desired_encoding="bgr8")
            rospy.logdebug("ROS Image converted for processing")
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

        tic = time.time()
        boxes, confs, clss = self.trt_yolo.detect1(cv_img, self.conf_th)
        cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
        toc = time.time()
        fps = 1.0 / (toc - tic)
        print("Cam1: " + str(fps) + "\n")
        self.publisher(boxes, confs, clss, "Cam1")
        
        if self.show_img:
            cv_img = show_fps(cv_img, fps)
            cv2.imshow("Cam 1", cv_img)
            cv2.waitKey(1)
        
        # converts back to ros_img type for publishing
        try:
            overlay_img = self.bridge.cv2_to_imgmsg(
                cv_img, encoding="passthrough")
            rospy.logdebug("CV Image converted for publishing")
            self.overlay_pub.publish(overlay_img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

    def img_callback2(self, ros_img):
        """Continuously capture images from camera and do object detection """

        # converts from ros_img to cv_img for processing
        # time both cv_bridge fns
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img, desired_encoding="bgr8")
            rospy.logdebug("ROS Image converted for processing")
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

        tic = time.time()
        boxes, confs, clss = self.trt_yolo.detect2(cv_img, self.conf_th)
        cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
        toc = time.time()
        fps = 1.0 / (toc - tic)
        print("Cam2: " + str(fps) + "\n")
        self.publisher(boxes, confs, clss, "Cam2")

        # time 1
        """
        if self.show_img:
            cv_img = show_fps(cv_img, fps)
            cv2.imshow("Cam 2", cv_img)
            # time 2
            cv2.waitKey(1)
        """
        
        # converts back to ros_img type for publishing
        try:
            overlay_img = self.bridge.cv2_to_imgmsg(
                cv_img, encoding="passthrough")
            rospy.logdebug("CV Image converted for publishing")
            self.overlay_pub.publish(overlay_img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

    def img_callback3(self, ros_img):
        """Continuously capture images from camera and do object detection """

        # converts from ros_img to cv_img for processing
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img, desired_encoding="bgr8")
            rospy.logdebug("ROS Image converted for processing")
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

        tic = time.time()
        boxes, confs, clss = self.trt_yolo.detect3(cv_img, self.conf_th)
        cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
        toc = time.time()
        fps = 1.0 / (toc - tic)
        print("Cam3: " + str(fps) + "\n")
        self.publisher(boxes, confs, clss, "Cam3")
        """
            if self.show_img:
                cv_img = show_fps(cv_img, fps)
                cv2.imshow("Cam 3", cv_img)
                cv2.waitKey(1)
        """
        # converts back to ros_img type for publishing
        try:
            overlay_img = self.bridge.cv2_to_imgmsg(
                cv_img, encoding="passthrough")
            rospy.logdebug("CV Image converted for publishing")
            self.overlay_pub.publish(overlay_img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

    def img_callback4(self, ros_img):
        """Continuously capture images from camera and do object detection """

        # converts from ros_img to cv_img for processing
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img, desired_encoding="bgr8")
            rospy.logdebug("ROS Image converted for processing")
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

        tic = time.time()
        boxes, confs, clss = self.trt_yolo.detect4(cv_img, self.conf_th)
        cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
        toc = time.time()
        fps = 1.0 / (toc - tic)
        print("Cam4: " + str(fps) + "\n")
        self.publisher(boxes, confs, clss, "Cam4")
        """
            if self.show_img:
                cv_img = show_fps(cv_img, fps)
                cv2.imshow("Cam 3", cv_img)
                cv2.waitKey(1)
        """
        # converts back to ros_img type for publishing
        try:
            overlay_img = self.bridge.cv2_to_imgmsg(
                cv_img, encoding="passthrough")
            rospy.logdebug("CV Image converted for publishing")
            self.overlay_pub.publish(overlay_img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

    def publisher(self, boxes, confs, clss, frame):
        """ Publishes to detector_msgs

        Parameters:
        boxes (List(List(int))) : Bounding boxes of all objects
        confs (List(double))	: Probability scores of all objects
        clss  (List(int))	: Class ID of all classes
        """
        detection2d = Detector2DArray()
        detection = Detector2D()
        detection2d.header.stamp = rospy.Time.now()
        detection2d.header.frame_id = "camera"  # change accordingly

        for i in range(len(boxes)):
            # boxes : xmin, ymin, xmax, ymax
            for _ in boxes:
                detection.header.stamp = rospy.Time.now()
                detection.header.frame_id = frame  # change accordingly
                detection.results.id = clss[i]
                detection.results.score = confs[i]

                detection.bbox.center.x = boxes[i][0] + \
                    (boxes[i][2] - boxes[i][0])/2
                detection.bbox.center.y = boxes[i][1] + \
                    (boxes[i][3] - boxes[i][1])/2
                detection.bbox.center.theta = 0.0  # change if required

                detection.bbox.size_x = abs(boxes[i][0] - boxes[i][2])
                detection.bbox.size_y = abs(boxes[i][1] - boxes[i][3])

            detection2d.detections.append(detection)

        self.detection_pub.publish(detection2d)


def main():
    yolo = yolov4()
    rospy.init_node('yolov4_trt_ros', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        del yolo
        rospy.on_shutdown(yolo.clean_up())
        print("Shutting down")


if __name__ == '__main__':
    main()
