# Python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import time
# Import from your code
import sys
from collections import deque
import heapq
sys.path.append('.')
import interpolation_model.config as cfg
from interpolation_model.Trainer import Model
from interpolation_model.benchmark.utils.padder import InputPadder

class InterpolationNode(Node):
    def __init__(self):
        super().__init__('interpolation_node')
        self.publisher_ = self.create_publisher(Image, '/decoder_interpolated_output', 10)
        self.publisher_2 = self.create_publisher(Image, '/decoder_combined_output', 10)
        self.subscription_1 = self.create_subscription(Image,'/decoder0_output',self.image_callback,10)
        self.subscription_2 = self.create_subscription(Image,'/decoder1_output',self.image_callback,10)
        
        self.bridge = CvBridge()

        # Model setting
        self.TTA = True
        model_name = 'ours_t'  # Set your model name here
        if model_name == 'ours_small_t':
            self.TTA = False
            cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
            cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
                F = 16,
                depth = [2, 2, 2, 2, 2]
            )
        else:
            cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
            cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
                F = 32,
                depth = [2, 2, 2, 4, 4]
            )
        self.model = Model(-1)
        self.model.load_model()
        self.model.eval()
        self.model.device()

        self.frames_queue = []
        self.sorted_frames_queue = []
        self.processing = False
        self.n = 8 # Set your n value here
        print("Interpolator Initialized")
        

    def image_callback(self, msg):
        print(int(msg.header.frame_id))
        img = self.bridge.imgmsg_to_cv2(msg)
        heapq.heappush(self.frames_queue,(int(msg.header.frame_id), img))
        if len(self.frames_queue) >= 2 and not self.processing:
            self.process_frames()

    def process_frames(self):
        self.processing = True

        frame1 = heapq.heappop(self.frames_queue)
        frame2 = self.frames_queue[0]
        print("Queue",[x[0] for x in self.frames_queue])
        print(frame1[0],frame2[0])
        
        if True:

            I0 = self._preprocess_image(frame1[1])
            I2 = self._preprocess_image(frame2[1])

            I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
            I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

            padder = InputPadder(I0_.shape, divisor=32)
            I0_, I2_ = padder.pad(I0_, I2_)

            images = [I0[:, :, ::-1]]
            img_msg = self.bridge.cv2_to_imgmsg(cv2.convertScaleAbs(frame1[1]), encoding='rgb8')
            self.publisher_.publish(img_msg)
            # st = time.time()
        
            for i in range(self.n - 1):
                pred = self.model.inference(I0_, I2_, TTA=self.TTA, timestep=(i+1)*(1./self.n), fast_TTA=self.TTA)
                images.append((padder.unpad(pred).detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1])
                img_msg = self.bridge.cv2_to_imgmsg(cv2.convertScaleAbs(images[-1]), encoding='bgr8')
                self.publisher_.publish(img_msg)
            # ed = time.time()
            # print("inference take: ", ed - st)
            images.append(I2[:, :, ::-1])
            img_msg = self.bridge.cv2_to_imgmsg(cv2.convertScaleAbs(frame1[1]), encoding='rgb8')
            self.publisher_2.publish(img_msg)
        self.processing = False

    def _preprocess_image(self, img):
        return img

def main(args=None):
    rclpy.init(args=args)
    interpolation_node = InterpolationNode()
    rclpy.spin(interpolation_node)
    interpolation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()