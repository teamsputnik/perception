
import time
import rclpy
import cv2
import torch as th
import numpy as np


from audio2photoreal.render_codes import BodyRenderer
from audio2photoreal.data_loaders.get_data import load_local_data

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sputnik_msgs.msg import Float64MultiArrayStamped


class DecoderNode(Node):
    def __init__(self):
        super().__init__('decoder_node')
        self.subscription = self.create_subscription(
            Float64MultiArrayStamped,
            '/lbs_keypoints',
            self.listener_callback,
            100)
        
        self.publisher_ = self.create_publisher(Image, '/decoder1_output', 10)
        self.bridge = CvBridge()
        
        # Initialize the body renderer and other necessary variables
        self.body_renderer = BodyRenderer(config_base="/home/user/sputnik_ws/src/perception/perception/audio2photoreal/dataset/PXB184", render_rgb=True, gpu=1).to("cuda:1")
        self.default_inputs = th.load(f"/home/user/sputnik_ws/src/perception/perception/audio2photoreal/assets/render_defaults_PXB184.pth")
        self.campos = self.default_inputs["campos"]
        self.Rt = self.default_inputs["Rt"]
        self.default_inputs["device"] = 'cuda:1'
        self.data_root = "/home/user/sputnik_ws/src/perception/perception/audio2photoreal/dataset/PXB184/PXB184"
        self.data_dict = load_local_data(self.data_root, audio_per_frame=1600)
        self.device = 'cuda1'

    def listener_callback(self, msg):
        data_pred = np.array(msg.data)
        frame_number = msg.header.frame_id
        
        if(int(frame_number) % 8 != 0):
            return 
        
        data_pred = np.expand_dims(data_pred,axis = 0)
        curr_face_chunk = self.data_dict["face"][0][5:5+data_pred.shape[0]]
        print(frame_number)
        # Run the function using the provided template
        # st = time.time()
        rgb = self.body_renderer.render_frame(data_pred, curr_face_chunk, self.Rt, self.campos)
        # ed = time.time()
        
        try:
            ros_image = self.bridge.cv2_to_imgmsg(rgb, "rgb8")
            ros_image.header.frame_id = frame_number
            # print(frame_number)
        except CvBridgeError as e:
            self.get_logger().warn('Failed to convert frame to ROS image: %s' % e)
            return

        print("-------------DECODER1 PUBLISHING--------------")
        self.publisher_.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)
    decoder_node = DecoderNode()
    rclpy.spin(decoder_node)
    decoder_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()