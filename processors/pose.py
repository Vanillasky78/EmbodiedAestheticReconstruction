"""
processors/pose.py
----------------------------------------
Pose estimation utilities:
- PoseEstimator: runs YOLOv8-Pose to get keypoints
- draw_skeleton_image: renders a clean skeleton silhouette
- make_pose_detector: adapter so camera.py can query motion

Requires:
    pip install ultralytics opencv-python pillow torch
    and download a YOLOv8 pose weights file (e.g. yolov8n-pose.pt)
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO


class PoseEstimator:
    def __init__(self, weights_path="yolov8n-pose.pt", device="cpu"):
        """
        weights_path: path to a YOLOv8 pose model, e.g. 'yolov8n-pose.pt'
        device: 'cuda', 'mps', or 'cpu'
        """
        self.model = YOLO(weights_path)
        self.device = device

    def extract_keypoints(self, frame_bgr):
        """
        Run pose detection on a single frame (BGR image from OpenCV).
        Returns:
            keypoints: list of (x, y) floats for the first detected person,
                       or None if no person found.
        """
        # YOLO expects RGB by default; but ultralytics will internally handle BGR fine.
        # We'll still just pass frame_bgr as-is.
        results = self.model.predict(
            frame_bgr,
            device=self.device,
            verbose=False,
        )

        if len(results) == 0:
            return None

        first_result = results[0]

        if first_result.keypoints is None or len(first_result.keypoints) == 0:
            return None

        # keypoints.xy shape: [num_people, num_points, 2]
        kps = first_result.keypoints.xy[0].cpu().numpy()  # take first person
        keypoints = [(float(x), float(y)) for (x, y) in kps]
        return keypoints

    def draw_skeleton_image(self, frame_bgr, keypoints, save_path):
        """
        Render a black background + white pose skeleton image
        derived from keypoints and save it.

        This "pose silhouette image" is what we later feed into OpenCLIP
        to embed the participant's pose.

        Returns:
            save_path (string)
        """
        if keypoints is None:
            raise ValueError("No keypoints provided to draw_skeleton_image().")

        H, W, _ = frame_bgr.shape
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # A simple skeleton connectivity list (COCO-ish style).
        # Index mapping depends on model; YOLOv8-pose typically uses:
        # 0:nose, 1:eye_l, 2:eye_r, 3:ear_l, 4:ear_r,
        # 5:shoulder_l, 6:shoulder_r,
        # 7:elbow_l, 8:elbow_r,
        # 9:wrist_l, 10:wrist_r,
        # 11:hip_l, 12:hip_r,
        # 13:knee_l, 14:knee_r,
        # 15:ankle_l, 16:ankle_r
        skeleton_pairs = [
            (5, 7), (7, 9),     # left arm
            (6, 8), (8, 10),    # right arm
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16), # right leg
            (5, 6), (5, 11), (6, 12), (11, 12),  # torso
            (5, 0), (6, 0)      # shoulders to head-ish
        ]

        # Draw joints as small circles.
        for (x, y) in keypoints:
            cv2.circle(canvas, (int(x), int(y)), 4, (255, 255, 255), -1)

        # Draw bones as lines.
        for (i, j) in skeleton_pairs:
            if i < len(keypoints) and j < len(keypoints):
                xi, yi = keypoints[i]
                xj, yj = keypoints[j]
                cv2.line(
                    canvas,
                    (int(xi), int(yi)),
                    (int(xj), int(yj)),
                    (255, 255, 255),
                    2
                )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, canvas)
        return save_path


def make_pose_detector(pose_estimator):
    """
    Returns a lightweight function that camera.py can call every frame.

    The returned function takes a BGR frame and returns a list of (x, y)
    keypoints for the first detected person, or None.
    """
    def detector(frame_bgr):
        return pose_estimator.extract_keypoints(frame_bgr)
    return detector
