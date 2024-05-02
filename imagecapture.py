#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__('image_capture_node')
        self.publisher_pose2D = self.create_publisher(Image, '/capture_image', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture("/dev/v4l/by-id/usb-UltraSemi_USB3_Video_20210623-video-index0")
        # self.cap = cv2.VideoCapture("/home/share/audio2photoreal/saksham2.mp4")
        #self.cap = cv2.VideoCapture("/home/share/audio2photoreal/annanya.avi")

        self.frames = []
        self.saved = False
        self.frame_count = 0
        self.timer = self.create_timer(1/32, self.timer_callback)  # Adjust timer as needed

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            print("no frame captured")
            self.get_logger().warn('No frame captured.')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # For looping the playback video comment for the stream
            return
        try:
            w, h = frame.shape[0], frame.shape[1]
            print("frame number:", self.frame_count)
            ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            ros_image.header.frame_id = str(self.frame_count)
        except CvBridgeError as e:
            self.get_logger().warn('Failed to convert frame to ROS image: %s' % e)
            return

        # Publish to pose2Dnode
        self.publisher_pose2D.publish(ros_image)
        self.frame_count += 1

def main(args=None):
    rclpy.init(args=args)
    image_capture_node = ImageCaptureNode()
    rclpy.spin(image_capture_node)

    image_capture_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()