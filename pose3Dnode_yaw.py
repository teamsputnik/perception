import os
import time
import cv2
import numpy as np
import rclpy
# from mmdeploy_runtime import PoseTracker
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from rnn_model_simple import RNN
from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding
from std_msgs.msg import Float64MultiArray, Float64
from sputnik_msgs.msg import Float64MultiArrayStamped
from sklearn.neighbors import KDTree
import sys
import cv2
import multiprocessing

sys.path.append("/home/share/audio2photoreal/")
sys.path.append("/home/share/audio2photoreal/PoseFormerV2")
sys.path.append("/home/share/audio2photoreal/PoseFormerV2/demo")

from PoseFormerV2.demo.vis import Pose2DEstimator, Pose3DEstimator, get_pose2D

VISUALIZATION_CFG = dict(
    coco=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
        palette=[(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
                 (255, 153, 255), (153, 204, 255), (255, 102, 255),
                 (255, 51, 255), (102, 178, 255), (51, 153, 255),
                 (255, 153, 153), (255, 102, 102), (255, 51, 51),
                 (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
                 (0, 0, 255), (255, 0, 0), (255, 255, 255)],
        link_color=[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ],
        point_color=[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ]),
    coco_wholebody=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                  (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
                  (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                  (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
                  (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
                  (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
                  (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
                  (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
                  (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
                  (129, 130), (130, 131), (131, 132)],
        palette=[(51, 153, 255), (0, 255, 0), (255, 128, 0), (255, 255, 255),
                 (255, 153, 255), (102, 178, 255), (255, 51, 51)],
        link_color=[
            1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1,
            1, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2,
            2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1,
            1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068,
            0.066, 0.066, 0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043,
            0.040, 0.035, 0.031, 0.025, 0.020, 0.023, 0.029, 0.032, 0.037,
            0.038, 0.043, 0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012,
            0.012, 0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007, 0.007,
            0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011, 0.009, 0.011,
            0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010, 0.034, 0.008,
            0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009, 0.009,
            0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01,
            0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
            0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032,
            0.02, 0.019, 0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047,
            0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
            0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031
        ]))

class Pose3DNode(Node):
    def __init__(self):
        super().__init__('lbs_node')
        timer_period = 0.1  # seconds
        
        self.model = RNN(357,1024,98).to("cuda:0")
        self.model.load_state_dict(torch.load("/home/user/saksham/regressor_model/checkpoints/rnn_model_simple_3Dpose_har3_full_dataset.pth"))
        self.model.eval()

        # ROS setup
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image,'/capture_image',self.image_callback,10)
        self.keypoints_pub = self.create_publisher(Float64MultiArrayStamped,'/lbs_keypoints',10)
        self.position_pub = self.create_publisher(Float64MultiArray,'/human_position',10)
        self.first_six_lbs = np.array([0.8238490223884583, -11.23712158203125, -27.187618255615234, 2.974771022796631, 0.06914862990379333, -0.1998562067747116])
        self.pred_pose = []
        self.pred_lbs_points = []
        self.frame_counter = 0
        self.interpolate_frame = 8 
        self.lbs_actual = self.ConcatFiles()
        self.tree = KDTree(self.lbs_actual)
        self.pose2d_model = Pose2DEstimator("cuda:0")
        self.pose3d_model = Pose3DEstimator("cuda:0")


    def ConcatFiles(self):
        concat_vector = []
        folder_path = "/home/share/audio2photoreal/output/dataset/lbs_gt"
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path,file_name)
            np_array = np.load(file_path)
            np_array = np.transpose(np_array,(1,2,0))
            concat_vector.extend(np_array)
        
        concat_vector = np.array(concat_vector).squeeze(2)
        return concat_vector


    def pose_publisher(self,poses,frame_number):
        msg = Float64MultiArrayStamped()
        if (np.shape(poses)[0]==0):
            self.get_logger().info("-----------NO HUMAN DETECTED----------")
        else:
            msg.data = poses.flatten().tolist()
            msg.header.frame_id = str(frame_number)
            self.keypoints_pub.publish(msg)

    def position_publisher(self,positions,image_size,frame_number):
        msg = Float64MultiArray()
        if (np.shape(positions)[0]==0):
            self.get_logger().info("-----------NO HUMAN DETECTED----------")
        else:
            msg.data = positions.flatten().tolist()
            msg.data.extend(image_size)
            self.position_pub.publish(msg)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame_number = int(msg.header.frame_id)
            if (frame_number % self.interpolate_frame==0):
                # self.get_logger().info(str(frame_number))
                keypoints = self.pose2d_model.get_pose2D_single_frame(frame)
                pred_pose = self.pose3d_model.predict(frame, keypoints)
                if pred_pose is not None:
                    self.pred_lbs_points = self.getLBSPoints(pred_pose)
                else:
                    self.pred_lbs_points = []
                self.position_publisher(keypoints,frame.shape,frame_number)
                self.pose_publisher(self.pred_lbs_points,frame_number)


        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")



    def getLBSPoints(self, frame):
        input_tensor = torch.tensor(frame)
        selected_data = input_tensor
        
        he = HarmonicEmbedding(3)
        input_data = selected_data.reshape(1, -1)
        input_data_embedding = he.forward(input_data).to(torch.float32).to("cuda:0").reshape(1, 1, -1)
        steps = input_data.shape[0] 
        pred_output = self.model(input_data_embedding.to(torch.float32).to("cuda:0"), steps).cpu().detach().numpy()
        pred_output = pred_output[:,-1]
        data_pred_first = np.zeros_like(pred_output)[..., :6]
        data_pred_first[:] = self.first_six_lbs
        pred_output = np.concatenate([data_pred_first, pred_output], axis=-1)
        # self.get_logger().info("-------------LBS CALCULATED-----------")    
        return pred_output

    def visualize(self, frame, results, thr=0.5, resize=1280, skeleton_type='coco'):
        skeleton = VISUALIZATION_CFG[skeleton_type]['skeleton']
        palette = VISUALIZATION_CFG[skeleton_type]['palette']
        link_color = VISUALIZATION_CFG[skeleton_type]['link_color']
        point_color = VISUALIZATION_CFG[skeleton_type]['point_color']

        scale = resize / max(frame.shape[0], frame.shape[1])
        keypoints, bboxes, _ = results
        
        scores = keypoints[..., 2]
        keypoints = (keypoints[..., :2] * scale).astype(int)
        bboxes *= scale
        img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        for kpts, score, bbox in zip(keypoints, scores, bboxes):
            show = [1] * len(kpts)
            for (u, v), color in zip(skeleton, link_color):
                if score[u] > thr and score[v] > thr:
                    cv2.line(img, kpts[u], tuple(kpts[v]), palette[color], 1, cv2.LINE_AA)
                else:
                    show[u] = show[v] = 0
            for kpt, show, color in zip(kpts, show, point_color):
                if show:
                    cv2.circle(img, kpt, 1, palette[color], 2, cv2.LINE_AA)
        return keypoints, scores, bboxes

def main(args=None):
    rclpy.init(args=args)
    pose_2d_node = Pose3DNode()
    rclpy.spin(pose_2d_node)
    pose_2d_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
