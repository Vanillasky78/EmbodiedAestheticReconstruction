import time
import numpy as np
import os, cv2, av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from processors.live_pose import LivePose

st.markdown("### 🎥 Live Mirror + Pose (Auto-capture on stillness)")

# ---- 控件
with st.expander("Live settings", expanded=True):
    conf = st.slider("Pose confidence", 0.05, 0.9, 0.4, 0.05)
    mirror = st.checkbox("Mirror preview", value=True)
    still_seconds = st.number_input("Stillness required (seconds)", 1.0, 10.0, 5.0, 0.5)  # 已改为 5s 默认
    sensitivity = st.slider("Motion sensitivity (lower = 更敏感)", 0.5, 5.0, 2.0, 0.1)
    manual_col = st.empty()

# ---- 状态
if "live_pose" not in st.session_state:
    st.session_state.live_pose = LivePose(yolo_weights="yolov8n-pose.pt", conf=conf)
else:
    st.session_state.live_pose.conf = conf

# 供 stillness 检测使用的状态
state = st.session_state
state.prev_kps = state.get("prev_kps", None)          # 上一帧关键点
state.last_move_ts = state.get("last_move_ts", time.time())
state.last_saved_path = state.get("last_saved_path", None)
state.auto_saved_flag = state.get("auto_saved_flag", False)

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

def _overlay_progress(img: np.ndarray, remain: float, total: float):
    """在画面底部绘制进度条 + 文本"""
    h, w = img.shape[:2]
    pad = 12
    bar_w = int(w * 0.6)
    bar_h = 10
    x0 = (w - bar_w) // 2
    y0 = h - pad - bar_h
    # 背景槽
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
    # 已静止时长
    elapsed = max(0.0, total - max(0.0, remain))
    frac = float(np.clip(elapsed / total, 0, 1))
    cv2.rectangle(img, (x0, y0), (x0 + int(bar_w * frac), y0 + bar_h), (0, 200, 0), -1)
    # 文本
    txt = f"Hold still: {remain:.1f}s"
    cv2.putText(img, txt, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 220, 10), 2, cv2.LINE_AA)

def _kps_motion(prev_kps, cur_kps, norm_scale):
    """
    计算关键点间的平均位移（归一化）。norm_scale 用画面对角线像素；sensitivity 越小越敏感。
    只在两帧都有的关键点计算。
    """
    if prev_kps is None or cur_kps is None:
        return None
    mask = (prev_kps[:, 0] > 0) & (prev_kps[:, 1] > 0) & (cur_kps[:, 0] > 0) & (cur_kps[:, 1] > 0)
    if mask.sum() < 4:
        return None
    d = np.linalg.norm(prev_kps[mask] - cur_kps[mask], axis=1).mean()
    return d / (norm_scale + 1e-6)

def _save_frame(bgr: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bgr)

def _video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    bgr = frame.to_ndarray(format="bgr24")

    # 姿态推理 + 骨骼绘制，拿到关键点
    vis, cur_kps = state.live_pose.infer_and_draw(bgr, mirror=mirror, return_kps=True)

    # 运动检测（基于关键点位移）
    h, w = vis.shape[:2]
    diag = float(np.hypot(h, w))  # 归一化尺度：画面对角线
    motion = _kps_motion(state.prev_kps, cur_kps, norm_scale=diag)
    now = time.time()

    # 阈值：以 sensitivity 转换为“允许的归一化位移”
    # sensitivity 越小阈值越小 -> 更容易判定为“在动”
    motion_thresh = 0.002 * sensitivity  # 经验值：0.002 对应轻微抖动

    if motion is None or motion > motion_thresh:
        state.last_move_ts = now
        state.auto_saved_flag = False  # 一旦动了，下次可再次触发抓拍

    remain = max(0.0, still_seconds - (now - state.last_move_ts))
    _overlay_progress(vis, remain, still_seconds)

    # 满足静止 5s 自动抓拍
    if remain <= 0 and (not state.auto_saved_flag):
        save_path = "data/interim/locked_frame.jpg"
        # 保存叠加骨骼的画面（镜像一致）
        _save_frame(vis, save_path)
        state.last_saved_path = save_path
        state.auto_saved_flag = True
        # 在画面上打个 “Saved!” 提示
        cv2.putText(vis, "Saved!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 240, 0), 3, cv2.LINE_AA)

    state.prev_kps = cur_kps
    return LivePose.avframe_from_bgr(vis)

webrtc_ctx = webrtc_streamer(
    key="live-pose-mirror-stillness",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=_video_frame_callback,
)

# 手动保存按钮（可选）
if manual_col.button("📸 Save frame (manual)"):
    if webrtc_ctx and webrtc_ctx.video_receiver:
        frame = webrtc_ctx.video_receiver.get_frame(timeout=2.0)
        bgr = frame.to_ndarray(format="bgr24")
        vis, _ = state.live_pose.infer_and_draw(bgr, mirror=mirror, return_kps=True)
        save_path = "data/interim/locked_frame.jpg"
        _save_frame(vis, save_path)
        state.last_saved_path = save_path
        st.success(f"Saved → {save_path}")
    else:
        st.error("No video frame yet — please allow camera and wait a second.")

# 保存结果提示
if state.last_saved_path:
    st.caption(f"Last saved: {state.last_saved_path}")
