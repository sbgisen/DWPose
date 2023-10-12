#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright (c) 2023 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import typing

import cv2
import cv_bridge
import message_filters
import numpy as np
import onnxruntime as ort
import rospkg
import rospy
import tf
from dwpose.onnxpose import inference_pose
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import RectArray
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image


class Inference(object):

    def __init__(self) -> None:
        path = rospkg.RosPack().get_path('dwpose')
        # self.classifier_name = rospy.get_param('~classifier_name', rospy.get_name())
        self.model_path = rospy.get_param('~model_path', path + '/config/dw-ll_ucoco_384.onnx')
        # onnx_det = path + '/ControlNet-v1-1-nightly/annotator/ckpts/yolox_l.onnx'

        self.face_model = rospy.get_param('~face_model_path', path + '/config/model.txt')
        self.face_points_68 = self._get_full_model_points(self.face_model)
        self.r_vec = None
        self.t_vec = None
        self.camera_info = rospy.wait_for_message('~camera_info', CameraInfo)
        self.__k = np.array(self.camera_info.K).reshape(3, 3)
        self.__d = np.array(self.camera_info.D)

        # self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=self.model_path, providers=['CUDAExecutionProvider'])
        self.sub_img = message_filters.Subscriber('~image', Image)
        self.sub_rect = message_filters.Subscriber('~rects', RectArray)
        self.sub_class = message_filters.Subscriber('~class', ClassificationResult)
        self.sync = message_filters.TimeSynchronizer([self.sub_img, self.sub_rect, self.sub_class], 20)
        self.sync.registerCallback(self.callback)
        self.bridge = cv_bridge.CvBridge()

        self.pub_viz = rospy.Publisher('~output/viz', Image, queue_size=1)
        self.pub_pose = rospy.Publisher('~output/pose', PoseArray, queue_size=1)

    def callback(self, img_msg: Image, rect_msg: RectArray, class_msg: ClassificationResult) -> None:
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')

        det_result = np.array([[r.x, r.y, r.x + r.width, r.y + r.height]
                               for r, c in zip(rect_msg.rects, class_msg.label_names) if c == 'person'])
        if len(det_result) == 0:
            return
        # det_result = inference_detector(self.session_det, img)
        keypoints, scores = inference_pose(self.session_pose, det_result, img)

        # keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        # neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # # neck score when visualizing pred
        # neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        # new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        # mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        # openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        # new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        # keypoints_info = new_keypoints_info

        # keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]
        res = self.visualize(img, keypoints, scores)
        self.pub_viz.publish(self.bridge.cv2_to_imgmsg(res, encoding='rgb8'))
        ret = PoseArray()
        ret.header = img_msg.header
        for keypoint in keypoints:
            face = Pose()
            r, t = self.solve(keypoint[23:91])
            face.position.x = t[0] / 1000
            face.position.y = t[1] / 1000
            face.position.z = t[2] / 1000
            q = tf.transformations.quaternion_from_euler(r[0], r[1], r[2])
            face.orientation.x = q[0]
            face.orientation.y = q[1]
            face.orientation.z = q[2]
            face.orientation.w = q[3]
            ret.poses.append(face)
        self.pub_pose.publish(ret)

    def visualize(self, img: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, thr: float = 0.3) -> np.ndarray:
        """Visualize the keypoints and skeleton on image.

        Args:
            img (np.ndarray): Input image in shape.
            keypoints (np.ndarray): Keypoints in image.
            scores (np.ndarray): Model predict scores.
            thr (float): Threshold for visualize.

        Returns:
            img (np.ndarray): Visualized image.
        """
        # default color
        skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
                    (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17), (15, 18), (15, 19),
                    (16, 20), (16, 21), (16, 22), (91, 92), (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                    (98, 99), (91, 100), (100, 101), (101, 102), (102, 103), (91, 104), (104, 105), (105, 106),
                    (106, 107), (91, 108), (108, 109), (109, 110), (110, 111), (112, 113), (113, 114), (114, 115),
                    (115, 116), (112, 117), (117, 118), (118, 119), (119, 120), (112, 121), (121, 122), (122, 123),
                    (123, 124), (112, 125), (125, 126), (126, 127), (127, 128), (112, 129), (129, 130), (130, 131),
                    (131, 132)]
        palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255], [255, 153, 255], [102, 178, 255],
                   [255, 51, 51]]
        link_color = [
            1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5,
            5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ]
        point_color = [
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
            1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ]

        # draw keypoints and skeleton
        for kpts, score in zip(keypoints, scores):
            for kpt, color in zip(kpts, point_color):
                cv2.circle(img, tuple(kpt.astype(np.int32)), 1, palette[color], 1, cv2.LINE_AA)
            for (u, v), color in zip(skeleton, link_color):
                if score[u] > thr and score[v] > thr:
                    cv2.line(img, tuple(kpts[u].astype(np.int32)), tuple(kpts[v].astype(np.int32)), palette[color], 2,
                             cv2.LINE_AA)

        return img

    def _get_full_model_points(self, filename: str) -> np.ndarray:
        """Get all 68 3D model points from file."""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1

        return model_points

    def solve(self, points: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Solve pose with all the 68 image points
        Args:
            points: points on image.

        Returns:
            (rotation_vector, translation_vector) as pose.
        """
        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(self.face_points_68, points, self.__k, self.__d)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(self.face_points_68,
                                                                points,
                                                                self.__k,
                                                                self.__d,
                                                                rvec=self.r_vec,
                                                                tvec=self.t_vec,
                                                                useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)


if __name__ == '__main__':
    rospy.init_node('pose_estimation')
    node = Inference()
    rospy.spin()
