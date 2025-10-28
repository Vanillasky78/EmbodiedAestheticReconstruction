"""
processors/camera.py
----------------------------------------
Camera loop that waits until the subject holds a pose
(with very little movement) for a given duration
(e.g. ~3 seconds), then captures and saves that frame.

This module does NOT draw skeletons. It only:
- reads webcam frames,
- checks motion using keypoints,
- writes the "locked" RGB frame to disk,
- returns the saved frame path.

Requires:
    pip install opencv-python numpy
"""

import os
import cv2
import time
import numpy as np
from datetime import datetime


def _avg_keypoint_movement(kpts_prev, kpts_curr):
    """
    Compute average Euclidean distance between previous keypoints and current keypoints.
    Returns a float (pixels). Lower = more stable.
    If we don't have both sets, return a large value so it's considered "not stable".
    """
    if kpts_prev is None or kpts_curr is None:
        return 1e9  # force "not stable" at start

    diffs = []
    for a, b in zip(kpts_prev, kpts_curr):
        if a is None or b is None:
            continue
        ax, ay = a
        bx, by = b
        diffs.append(np.linalg.norm(np.array([ax, ay]) - np.array([bx, by])))

    if len(diffs) == 0:
        return 1e9

    return float(sum(diffs) / len(diffs))


def capture_locked_frame(
    pose_detector,
    save_dir="data/interim/raw_frames",
    stable_seconds=3.0,
    movement_threshold=5.0,
    camera_index=0,
    show_debug_window=False
):
    """
    Continuously read frames from webcam.
    Use `pose_detector(frame_bgr)` to get body keypoints for the first person.
    Track how much those keypoints move from frame to frame.
    When movement stays below `movement_threshold` pixels
    for at least `stable_seconds` seconds,
    save that frame as the "locked frame" and return its path.

    Arguments:
        pose_detector: function(frame_bgr) -> list[(x,y), ...] or None
        save_dir: where to store the captured frame
        stable_seconds: how long the subject must stay still
        movement_threshold: how little they have to move (pixels avg)
        camera_index: which webcam to open (0 = default)
        show_debug_window: if True, show live feed with text overlay
                           (useful for development, not for installation)

    Returns:
        locked_path (str): path to the saved frame
    """

    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    prev_keypoints = None
    stable_start_time = None
    locked_path = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                continue

            # Get keypoints for current frame
            curr_keypoints = pose_detector(frame_bgr)

            # Compute movement
            movement = _avg_keypoint_movement(prev_keypoints, curr_keypoints)

            # Check stability
            is_still = movement < movement_threshold

            if is_still:
                # start timing if we just became still
                if stable_start_time is None:
                    stable_start_time = time.time()
                elapsed = time.time() - stable_start_time
            else:
                # reset timer
                stable_start_time = None
                elapsed = 0.0

            # Optional preview window for debugging
            if show_debug_window:
                debug_frame = frame_bgr.copy()
                text = (
                    f"movement={movement:.2f} px | "
                    f"still_for={elapsed:.1f}s / {stable_seconds:.1f}s"
                )
                color = (0, 255, 0) if is_still else (0, 0, 255)
                cv2.putText(
                    debug_frame,
                    text,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA
                )
                cv2.imshow("live", debug_frame)
                # ESC to break in dev mode
                if cv2.waitKey(1) == 27:
                    break

            # If we have stayed still long enough → lock frame
            if is_still and stable_start_time is not None:
                if elapsed >= stable_seconds:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    locked_path = os.path.join(save_dir, f"user_frame_{ts}.png")
                    cv2.imwrite(locked_path, frame_bgr)
                    break

            prev_keypoints = curr_keypoints

    finally:
        cap.release()
        if show_debug_window:
            cv2.destroyAllWindows()

    if locked_path is None:
        raise RuntimeError("No locked frame was captured.")
    return locked_path
