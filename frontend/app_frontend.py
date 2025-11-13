"""
app_frontend.py
------------------------------------
EAR — Skeleton Exhibition Frontend (API version)

Layout:
  - Left: live camera with YOLOv8-Pose skeleton overlay
  - Right: matched artwork image with simple labels
  - Full-bleed, symmetric left/right panels (no sidebar, no header bar)

Flow:
  1) Left panel uses WebRTC + YOLOv8-Pose to draw pose on the participant.
  2) When the participant holds still for ~2.5s, we auto-capture a frame.
  3) The captured frame is sent to the FastAPI backend `/match` endpoint.
  4) The backend returns the top-1 matched artwork (filename, artist, year, score).
  5) Right panel loads the local image file (data/local/images/<filename>)
     and overlays labels such as artist / year / (price_text if available).
"""

import io
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# --------------------- Paths & constants ---------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Use the new multi-museum layout: data/local/...
DATA_ROOT = PROJECT_ROOT / "data" / "local"
IMAGES_DIR = DATA_ROOT / "images"
# Optional CSV if you want to cross-check metadata locally
CSV_FILE = DATA_ROOT / "portrait_works.csv"

# FastAPI backend endpoint
API_URL = "http://127.0.0.1:8000/match"

# Stillness detection (front-end auto-capture)
STILLNESS_SEC = 2.5
FPS_ASSUMED = 15
MOTION_EPS_CXCY = 12.0
MOTION_EPS_AREA = 0.08

# WebRTC config (works for Safari as well)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Try YOLOv8-Pose
try:
    from ultralytics import YOLO

    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

# --------------------- Streamlit global UI setup ---------------------

st.set_page_config(page_title="EAR — Skeleton Frontend", layout="wide")

# Kiosk-style CSS: hide header/footer/sidebar, make two full-height panels
st.markdown(
    """
<style>
/* Hide Streamlit chrome */
header, footer, [data-testid="stToolbar"] {visibility: hidden;}
section[data-testid="stSidebar"] {display: none;}
div.block-container {padding: 0; margin: 0; max-width: 100%;}

/* Full-bleed 2-column layout */
.exhibit {
    display: grid;
    grid-template-columns: 1fr 1fr;
    height: 100vh;
    background: #000;
    position: relative;
}
.panel {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
}
.left  { background: #000; }
.right { background: #000; }

/* vertical divider */
.exhibit::before {
    content: "";
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 1px;
    background: #333;
}

/* Remove padding around images */
[data-testid="stImage"] img {
    object-fit: contain;
    width: 100%;
    height: 100%;
}

/* Hide default STOP button of WebRTC block */
button[kind="secondary"] {display: none !important;}
</style>
""",
    unsafe_allow_html=True,
)

# Render skeleton container div (for visual alignment)
st.markdown(
    '<div class="exhibit"><div class="panel left" id="left"></div>'
    '<div class="panel right" id="right"></div></div>',
    unsafe_allow_html=True,
)

# Use two Streamlit columns for content
left_col, right_col = st.columns(2)

# --------------------- Small utilities ---------------------


def load_font(size: int = 36) -> ImageFont.FreeTypeFont:
    """Try a few common fonts on macOS; fallback to default."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def overlay_labels(painting: Image.Image, meta: Dict) -> Image.Image:
    """
    Simple yellow labels on the left-middle:
      line 1: price_text (or "—")
      line 2: year
      line 3: artist
    """
    im = painting.convert("RGB").copy()
    draw = ImageDraw.Draw(im)

    yellow = (255, 235, 59)
    black = (0, 0, 0)
    font_big = load_font(44)
    font_small = load_font(36)

    price_text = meta.get("price_text") or "—"
    year_text = str(meta.get("year") or "—")
    artist = meta.get("artist") or "artist name"

    texts = [price_text, year_text, artist]
    fonts = [font_big, font_small, font_big]

    y = im.height // 2
    x = 24
    pad_x, pad_y = 16, 10

    for t, f in zip(texts, fonts):
        # measure text box
        l, t0, r, b = draw.textbbox((0, 0), t, font=f)
        w, h = r - l, b - t0

        draw.rectangle(
            [x, y, x + w + 2 * pad_x, y + h + 2 * pad_y],
            fill=yellow,
        )
        draw.text((x + pad_x, y + pad_y), t, fill=black, font=f)
        y += h + 30

    return im


def open_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def ensure_image_path_from_filename(filename: str) -> Optional[Path]:
    """Resolve filename to data/local/images/<filename>."""
    if not filename:
        return None
    p = Path(filename)
    if not p.is_absolute():
        p = IMAGES_DIR / p.name
    return p if p.exists() else None


def call_backend_match(image: Image.Image, museum: str = "local", topk: int = 3) -> Dict:
    """
    Send the captured image to FastAPI backend /match.

    Returns backend JSON:
      {
        "museum": ...,
        "topk": ...,
        "results": [
          {"filename":..., "title":..., "artist":..., "year":..., "score":...},
          ...
        ]
      }
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    files = {"image": ("capture.png", buf, "image/png")}
    data = {"museum": museum, "topk": str(topk)}

    resp = requests.post(API_URL, files=files, data=data, timeout=60)
    resp.raise_for_status()
    return resp.json()


# --------------------- Video Processor (left skeleton) ---------------------


class SkeletonProcessor(VideoProcessorBase):
    """
    - Draw YOLOv8-Pose skeleton on the left stream
    - Detect stillness and auto-capture a frame
    """

    def __init__(self):
        self.model = YOLO("yolov8n-pose.pt") if HAS_YOLO else None
        self.cx_buf = deque(maxlen=60)
        self.cy_buf = deque(maxlen=60)
        self.area_buf = deque(maxlen=60)
        self.last_start: Optional[float] = None

        self.captured: Optional[Image.Image] = None
        self.captured_ts: float = 0.0
        self.lock = threading.Lock()

    def _bbox_largest(self, rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if not self.model:
            return None
        res = self.model.predict(rgb, imgsz=640, device="cpu", verbose=False)
        if not len(res):
            return None
        boxes = res[0].boxes.xyxy
        if boxes is None or len(boxes) == 0:
            return None
        b = boxes.cpu().numpy()
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        i = int(np.argmax(areas))
        return b[i].astype(int).tolist()  # x1, y1, x2, y2

    def _update_stillness(self, bbox, w, h) -> bool:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        area = max(1.0, (x2 - x1) * (y2 - y1))
        rel_area = area / float(w * h)

        self.cx_buf.append(cx)
        self.cy_buf.append(cy)
        self.area_buf.append(rel_area)

        if len(self.cx_buf) < int(STILLNESS_SEC * FPS_ASSUMED * 0.6):
            self.last_start = None
            return False

        stdx = float(np.std(self.cx_buf))
        stdy = float(np.std(self.cy_buf))
        stda = float(np.std(self.area_buf))

        stable = (stdx < MOTION_EPS_CXCY) and (stdy < MOTION_EPS_CXCY) and (stda < MOTION_EPS_AREA)
        now = time.time()

        if stable:
            if self.last_start is None:
                self.last_start = now
            return (now - self.last_start) >= STILLNESS_SEC
        else:
            self.last_start = None
            return False

    def recv(self, frame):
        import av

        bgr = frame.to_ndarray(format="bgr24")
        rgb = bgr[:, :, ::-1]
        h, w, _ = rgb.shape

        if self.model:
            res = self.model.predict(rgb, imgsz=640, device="cpu", verbose=False)
            plotted = res[0].plot()  # BGR with skeleton overlay
            out_rgb = plotted[:, :, ::-1]
        else:
            out_rgb = rgb

        bbox = self._bbox_largest(rgb)
        if bbox and self._update_stillness(bbox, w, h):
            with self.lock:
                # small debounce (0.3s) to avoid multiple captures at once
                if time.time() > self.captured_ts + 0.3:
                    self.captured = Image.fromarray(rgb)
                    self.captured_ts = time.time()
                    self.last_start = None

        out_bgr = out_rgb[:, :, ::-1]
        return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")


# --------------------- Left: camera + skeleton ---------------------

with left_col:
    ctx = webrtc_streamer(
        key="ear-skeleton",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=SkeletonProcessor,
        async_processing=True,
    )

    if ctx and ctx.video_processor:
        st.session_state["proc"] = ctx.video_processor
        st.markdown(
            "<div style='position:absolute; top:16px; left:16px; color:#f5f5f5; "
            "font-size:16px; font-family:system-ui;'>"
            "Hold your pose for ~2–3 seconds to capture."
            "</div>",
            unsafe_allow_html=True,
        )

# --------------------- Right: matched artwork ---------------------

with right_col:
    placeholder = st.empty()

    proc: Optional[SkeletonProcessor] = st.session_state.get("proc")
    if not proc:
        placeholder.info("Waiting for camera…")
    else:
        # Auto-refresh to poll for new captures
        try:
            st.autorefresh(interval=700, limit=None, key="ear_frontend_poll")
        except Exception:
            pass

        with proc.lock:
            cap = proc.captured
            ts = proc.captured_ts

        last_ts = st.session_state.get("last_ts", 0.0)
        if cap is not None and ts > last_ts:
            # new capture → call backend
            try:
                resp = call_backend_match(cap, museum="local", topk=3)
                st.session_state["last_backend_json"] = resp
                st.session_state["last_ts"] = ts
            except Exception as e:
                st.session_state["last_backend_json"] = {
                    "error": f"Backend error: {type(e).__name__}: {e}"
                }
                st.session_state["last_ts"] = ts

        data = st.session_state.get("last_backend_json")
        if not data or "results" not in data:
            placeholder.info("Hold still in front of the camera to trigger matching…")
        elif "error" in data:
            placeholder.error(data["error"])
        else:
            results = data.get("results", [])
            if not results:
                placeholder.warning("No match results from backend.")
            else:
                # Take top-1 result
                top = results[0]
                filename = top.get("filename")
                artist = top.get("artist")
                year = top.get("year")
                score = top.get("score")

                img_path = ensure_image_path_from_filename(filename or "")
                if not img_path:
                    placeholder.error(
                        f"Matched filename not found locally: {filename}. "
                        f"Check data/local/images/."
                    )
                else:
                    painting = open_image(img_path)
                    if painting is None:
                        placeholder.error(f"Failed to open image: {img_path}")
                    else:
                        meta = {
                            "artist": artist,
                            "year": year,
                            # price_text is not yet returned by backend, so we use '—'
                            "price_text": None,
                        }
                        painted = overlay_labels(painting, meta)
                        caption = f"{artist or 'artist'} — {year or ''}  (score: {score:.3f})"
                        placeholder.image(painted, caption=caption, use_column_width=True)
