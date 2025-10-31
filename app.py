# app.py
# Streamlit live mirror with YOLOv8 pose overlay + stillness auto-capture + artwork retrieval
import os
import time
import json
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
import streamlit as st

# =============== Optional: use your repository classes if available ===============
# Retrieval matcher (FAISS + OpenCLIP) — expected in your repo
try:
    from processors.retrieval import Matcher  # ensure processors/retrieval.py defines class Matcher
except Exception as e:
    Matcher = None
    _matcher_import_err = e

# ---------------- Pose detector via ultralytics YOLOv8-Pose ----------------
def load_yolo_pose(model_path: str = "yolov8n-pose.pt"):
    """
    Lazy-load YOLOv8 pose model. Returns (model, last_error).
    """
    try:
        from ultralytics import YOLO
        if not os.path.exists(model_path):
            # allow ultralytics to download default if not present
            model = YOLO("yolov8n-pose.pt")
        else:
            model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, e


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, conf: float = 0.4) -> np.ndarray:
    """
    Draw YOLOv8-pose skeleton on frame.
    keypoints shape: (N, 17, 3) — (x, y, score)
    """
    vis = frame.copy()
    if keypoints is None or len(keypoints) == 0:
        return vis

    # Common COCO pairs for 17-keypoint set (YOLOv8)
    pairs = [
        (5, 7), (7, 9),     # left arm
        (6, 8), (8, 10),    # right arm
        (11, 13), (13, 15), # left leg
        (12, 14), (14, 16), # right leg
        (5, 6), (11, 12),   # shoulders, hips
        (5, 11), (6, 12)    # torso diagonals
    ]
    for person in keypoints:
        pts = person[:, :2].astype(int)
        scores = person[:, 2]
        # joints
        for i, (x, y) in enumerate(pts):
            if scores[i] >= conf:
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        # bones
        for a, b in pairs:
            if scores[a] >= conf and scores[b] >= conf:
                xa, ya = pts[a]
                xb, yb = pts[b]
                cv2.line(vis, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)
    return vis


# ---------------- Camera device helpers ----------------
def enumerate_devices(max_idx: int = 3) -> List[int]:
    """
    Try indices 0..max_idx to see which ones open successfully.
    """
    ok = []
    for i in range(max_idx + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok.append(i)
            cap.release()
    return ok


# ---------------- Stillness detector ----------------
class StillnessDetector:
    """
    Simple stillness by frame diff (lower=more sensitive).
    If avg abs diff < threshold for 'require_s' seconds -> trigger capture.
    """
    def __init__(self, sensitivity: float = 2.0, require_s: float = 5.0):
        # sensitivity slider (lower = 更敏感); map into pixel-diff threshold
        # pick threshold ~ (sensitivity * 2.5)
        self.threshold = max(0.5, sensitivity * 2.5)
        self.require_s = max(0.5, require_s)
        self.last_gray = None
        self.ok_since = None

    def update(self, frame_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Returns (is_still_now, diff_value)
        """
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        if self.last_gray is None:
            self.last_gray = g
            self.ok_since = None
            return False, 999.0

        diff = cv2.absdiff(g, self.last_gray)
        self.last_gray = g
        val = float(np.mean(diff))

        if val < self.threshold:
            if self.ok_since is None:
                self.ok_since = time.time()
            if time.time() - self.ok_since >= self.require_s:
                # satisfied
                self.ok_since = None  # reset to avoid repeated triggers
                return True, val
        else:
            self.ok_since = None

        return False, val


# ---------------- Persistence helpers ----------------
def save_locked_frame(frame_bgr: np.ndarray, out_dir: str = "data/interim") -> Tuple[bool, Optional[str]]:
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_dir, "locked_frame.jpg")
        ok = cv2.imwrite(out_path, frame_bgr)
        return (ok, out_path if ok else None)
    except Exception:
        return False, None


# ---------------- Retrieval display ----------------
@st.cache_resource
def _load_matcher_cached():
    if Matcher is None:
        raise RuntimeError(f"Cannot import Matcher: {_matcher_import_err}")
    device = "mps"  # change to "cpu"/"cuda" if needed
    idx_dir = "indexing"
    return Matcher(
        faiss_index_path=os.path.join(idx_dir, "faiss.index"),
        ids_path=os.path.join(idx_dir, "ids.npy"),
        meta_path=os.path.join(idx_dir, "meta.json"),
        images_dir="data/images",
        device=device,
    )


def run_and_show_matches(saved_path: str, top_k: int = 3):
    st.subheader("🎨 Top matches")
    try:
        matcher = _load_matcher_cached()
    except Exception as e:
        st.error(f"Matcher init failed: {e}")
        return

    try:
        results = matcher.search_file(saved_path, top_k=top_k)
    except Exception as e:
        st.error(f"Search failed: {e}")
        return

    if not results:
        st.warning("No results returned.")
        return

    cols = st.columns(len(results))
    for i, r in enumerate(results):
        # tolerate different keys
        fname = r.get("file_name") or r.get("image") or r.get("path")
        score = r.get("score")
        title = r.get("title_en") or r.get("title") or ""
        artist = r.get("artist_en") or r.get("artist") or ""

        img_path = os.path.join("data/images", fname) if fname and not os.path.isabs(fname) else fname
        caption = f"#{i+1}  {artist} — *{title}*"
        if score is not None:
            caption += f"\nscore: {score:.3f}"
        with cols[i]:
            if img_path and os.path.exists(img_path):
                st.image(img_path, use_container_width=True, caption=caption)
            else:
                st.write(caption)
                st.info(f"Image not found: {img_path}")


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Embodied Aesthetic Reconstruction", layout="wide")
st.title("🪞 Live Mirror + Pose (Auto-capture on stillness)")

with st.expander("Live settings", expanded=True):
    pose_conf = st.slider("Pose confidence", 0.1, 0.9, 0.40, 0.01)
    mirror_preview = st.checkbox("Mirror preview", value=True)
    still_seconds = st.number_input("Stillness required (seconds)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
    sens = st.slider("Motion sensitivity (lower = 更敏感)", 0.5, 5.0, 2.0, 0.1)

    col_btn = st.columns(2)
    with col_btn[0]:
        manual = st.button("📸 Save frame (manual)")
    with col_btn[1]:
        topk = st.slider("Top-K to display", 1, 6, 3)

# device selection
avail = enumerate_devices(4)
dev_idx = st.selectbox("Select camera device", options=avail if avail else [0], index=0, format_func=lambda x: f"Device {x}")

# preview area
frame_area = st.empty()
status_line = st.empty()

# Session flags
if "running" not in st.session_state:
    st.session_state.running = False
if "last_bgr" not in st.session_state:
    st.session_state.last_bgr = None

start_col, stop_col = st.columns(2)
with start_col:
    start = st.button("START", type="primary", disabled=st.session_state.running)
with stop_col:
    stop = st.button("STOP", disabled=not st.session_state.running)

# Manual save triggers retrieval
if manual and st.session_state.last_bgr is not None:
    ok, p = save_locked_frame(st.session_state.last_bgr)
    if ok:
        st.success(f"Saved: {p}")
        run_and_show_matches(p, top_k=topk)
    else:
        st.error("Save failed (no frame yet?)")

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# Main loop
if st.session_state.running:
    # init camera
    cap = cv2.VideoCapture(int(dev_idx))
    if not cap.isOpened():
        st.error(f"Could not open camera device {dev_idx}")
        st.session_state.running = False
    else:
        # init pose
        yolo, pose_err = load_yolo_pose("yolov8n-pose.pt")
        if yolo is None:
            st.warning(f"Pose not available: {pose_err}")

        detector = StillnessDetector(sensitivity=sens, require_s=still_seconds)
        fps_t0 = time.time()
        frames = 0

        # run loop
        while st.session_state.running and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                status_line.error("Capture failed.")
                break

            # mirror if chosen
            if mirror_preview:
                frame = cv2.flip(frame, 1)

            draw = frame.copy()

            # pose overlay
            if yolo is not None:
                try:
                    # YOLOv8 inference — return keypoints
                    res = yolo.predict(draw, verbose=False, conf=pose_conf)
                    kps = None
                    if res and len(res) > 0 and hasattr(res[0], "keypoints") and res[0].keypoints is not None:
                        # ultralytics Keypoints object -> .xy and .xyn .conf
                        # we want (N, 17, 3) : x,y,conf
                        kp = res[0].keypoints
                        xy = kp.xy[0].cpu().numpy() if hasattr(kp, "xy") else None
                        confs = kp.conf[0].cpu().numpy() if hasattr(kp, "conf") and kp.conf is not None else None
                        if xy is not None:
                            if confs is None:
                                confs = np.ones((xy.shape[0], 1), dtype=np.float32)
                            kps = np.concatenate([xy, confs], axis=1).reshape(1, -1, 3)
                    draw = draw_skeleton(draw, kps, conf=pose_conf)
                except Exception as e:
                    status_line.warning(f"Pose error: {e}")

            st.session_state.last_bgr = frame.copy()

            # stillness
            is_still, diff_val = detector.update(frame)
            msg = f"Δ={diff_val:.2f} | threshold≈{detector.threshold:.2f}"
            if is_still:
                ok, path = save_locked_frame(frame)
                if ok:
                    status_line.success(f"Auto-captured: {path}")
                    run_and_show_matches(path, top_k=topk)
                else:
                    status_line.error("Auto-save failed.")

            # FPS estimate
            frames += 1
            if frames % 10 == 0:
                now = time.time()
                fps = 10.0 / max(1e-6, (now - fps_t0))
                fps_t0 = now
                cv2.putText(draw, f"FPS: {fps:.1f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(draw, f"FPS: {fps:.1f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # show
            frame_area.image(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB), use_container_width=True)
            status_line.info(("STILL…" if detector.ok_since else "MOVE…") + "  |  " + msg)

            # allow UI to breathe
            # break quickly if STOP pressed
            if not st.session_state.running:
                break
            # tiny sleep to avoid CPU 100%
            time.sleep(0.01)

        cap.release()
        status_line.info("Camera released.")
else:
    # idle state: show placeholder
    with frame_area.container():
        st.caption("选择摄像头并点击 START 以启动实时镜像 + 骨骼叠加。")
