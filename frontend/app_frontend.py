from __future__ import annotations

import io
import os
import time
import threading
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
from streamlit_webrtc import (
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
    webrtc_streamer,
)

# Try import YOLO (pose)
try:
    from ultralytics import YOLO

    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

# =================== paths & constants ===================

FRONTEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONTEND_DIR.parent

# Global mixed data (local + met)
DATA_DIR = ROOT_DIR / "data" / "mixed"
IMAGES_DIR = DATA_DIR / "images"

META_CSV_CANDIDATES = [
    DATA_DIR / "embeddings_meta.csv",
    ROOT_DIR / "data" / "local" / "portrait_works_enhanced_english.csv",
    ROOT_DIR / "data" / "local" / "portrait_works.csv",
]

DEFAULT_API_URL = "http://127.0.0.1:8000/match"
APP_TITLE = "Embodied Aesthetic Reconstruction"

YOLO_MODEL_PATH = FRONTEND_DIR / "yolov8n-pose.pt"

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

STILLNESS_SEC = 3.5
MAX_BUF_SEC = 5.0
FPS_ASSUMED = 12
MOTION_EPS_CXCY = 4.0
MOTION_EPS_AREA = 0.03
MIN_FACE_AREA = 0.06

YELLOW = (255, 235, 59)
BLACK = (0, 0, 0)
HOT_PINK = (255, 30, 180)

RIGHT_IMG_MAXW = 900

# =================== utils ===================

_META_CACHE: Optional[Dict[str, Dict]] = None


def load_meta_mapping() -> Dict[str, Dict]:
    """Load local CSV and index by filename."""
    global _META_CACHE
    if _META_CACHE is not None:
        return _META_CACHE

    import csv

    rows: List[Dict] = []
    for p in META_CSV_CANDIDATES:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            break

    mapping: Dict[str, Dict] = {}
    for r in rows:
        fname = (
            r.get("filename")
            or r.get("image_path")
            or r.get("path")
            or r.get("file")
        )
        if fname:
            mapping[str(fname)] = r

    _META_CACHE = mapping
    return mapping


def lookup_meta(filename: str) -> Dict:
    mapping = load_meta_mapping()
    return mapping.get(filename, {})


def safe_open_image(p: Path) -> Optional[Image.Image]:
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def ensure_image_path(filename: str) -> Optional[Path]:
    if not filename:
        return None
    p = Path(filename)
    if not p.is_absolute():
        p = IMAGES_DIR / p
    return p if p.exists() else None


def _load_font(size: int = 40) -> ImageFont.FreeTypeFont:
    """Prefer Courier / Courier New, fallback to Arial / default."""
    candidates = [
        "/Library/Fonts/Courier New.ttf",
        "/System/Library/Fonts/Courier.dfont",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_tiny_metrics_top_right(
    im: Image.Image, lines: List[str], size: int = 16, margin: int = 10
) -> Image.Image:
    """Pink tiny text: pose metrics at top-right."""
    if not lines:
        return im
    img = im.copy()
    d = ImageDraw.Draw(img)
    font = _load_font(size)

    widths = []
    for s in lines:
        l, t, r, b = d.textbbox((0, 0), s, font=font)
        widths.append(r - l)
    wmax = max(widths) if widths else 0

    x = img.width - margin - wmax
    y = margin
    for s in lines:
        d.text((x, y), s, fill=HOT_PINK, font=font)
        _, _, _, b = d.textbbox((0, 0), s, font=font)
        y += int(b * 0.95)
    return img


def overlay_right_labels(im: Image.Image, meta: Dict) -> Image.Image:
    """
    Draw yellow price bar etc. on the matched artwork (Courier font).
    """
    img = im.copy()
    d = ImageDraw.Draw(img)
    font = _load_font(size=max(24, img.width // 20))

    price_text = meta.get("price_text") or ""
    year = str(meta.get("year") or "").strip()
    if not price_text and not year:
        return img

    padding_x = img.width * 0.03
    padding_y = img.height * 0.02
    bar_h = img.height * 0.10

    x0 = padding_x
    y0 = img.height - bar_h - padding_y
    x1 = img.width - padding_x
    y1 = img.height - padding_y

    d.rectangle([x0, y0, x1, y1], fill=YELLOW)

    text_lines: List[str] = []
    if price_text:
        text_lines.append(str(price_text))
    if year:
        text_lines.append(str(year))

    tx = x0 + padding_x
    ty = y0 + padding_y * 0.3
    for line in text_lines:
        d.text((tx, ty), line, fill=BLACK, font=font)
        _, _, _, b = d.textbbox((0, 0), line, font=font)
        ty += b * 1.1

    return img


def format_metrics(kps: Dict[int, Tuple[float, float] | None]) -> List[str]:
    """
    Build several pose-related text metrics for overlay & logging.
    Very lightweight heuristics, not scientific.
    """
    def _angle(a, b, c) -> Optional[float]:
        if a is None or b is None or c is None:
            return None
        ax, ay = a
        bx, by = b
        cx, cy = c
        v1 = np.array([ax - bx, ay - by], dtype=float)
        v2 = np.array([cx - bx, cy - by], dtype=float)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        cos = float(np.dot(v1, v2) / (n1 * n2))
        cos = max(-1.0, min(1.0, cos))
        return float(np.degrees(np.arccos(cos)))

    nose = kps.get(0)
    l_shoulder = kps.get(5)
    r_shoulder = kps.get(6)
    l_elbow = kps.get(7)
    r_elbow = kps.get(8)
    l_wrist = kps.get(9)
    r_wrist = kps.get(10)

    lines: List[str] = []

    # Head tilt (approx)
    if nose and l_shoulder and r_shoulder:
        sx = 0.5 * (l_shoulder[0] + r_shoulder[0])
        sy = 0.5 * (l_shoulder[1] + r_shoulder[1])
        dx = nose[0] - sx
        dy = nose[1] - sy
        if dx != 0 or dy != 0:
            tilt = np.degrees(np.arctan2(dy, dx))
            lines.append(f"Head tilt: {tilt:+.1f}°")

    # Shoulder angle
    if l_shoulder and r_shoulder:
        dx = r_shoulder[0] - l_shoulder[0]
        dy = r_shoulder[1] - l_shoulder[1]
        if dx != 0 or dy != 0:
            ang = np.degrees(np.arctan2(dy, dx))
            lines.append(f"Shoulder: {ang:+.1f}°")

    # Elbow angles
    ang_l = _angle(l_shoulder, l_elbow, l_wrist) if (l_shoulder and l_elbow and l_wrist) else None
    ang_r = _angle(r_shoulder, r_elbow, r_wrist) if (r_shoulder and r_elbow and r_wrist) else None
    if ang_l is not None:
        lines.append(f"L elbow: {ang_l:.0f}°")
    if ang_r is not None:
        lines.append(f"R elbow: {ang_r:.0f}°")

    return lines


# =================== WebRTC video processor ===================

class CuratorialProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        if HAS_YOLO:
            try:
                local = YOLO_MODEL_PATH
                self.model = YOLO(str(local if local.exists() else "yolov8n-pose.pt"))
            except Exception:
                self.model = None

        maxlen = max(6, int(MAX_BUF_SEC * FPS_ASSUMED))
        self.cx_buf = deque(maxlen=maxlen)
        self.cy_buf = deque(maxlen=maxlen)
        self.area_buf = deque(maxlen=maxlen)
        self.last_stable_ts: Optional[float] = None

        self.captured_img: Optional[Image.Image] = None
        self.captured_ts: float = 0.0
        self.captured_metrics: List[str] = []

        self.latest_rgb: Optional[np.ndarray] = None
        self.last_metrics_lines: List[str] = []

        self.lock = threading.Lock()

    def _detect_bbox(self, rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if not self.model:
            return None
        res = self.model.predict(rgb, imgsz=640, device="cpu", verbose=False)
        if len(res) == 0 or res[0].boxes is None:
            return None
        b = res[0].boxes.xyxy
        if b is None or len(b) == 0:
            return None
        b = b.cpu().numpy()
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        i = int(np.argmax(areas))
        x1, y1, x2, y2 = b[i].astype(int).tolist()
        return x1, y1, x2, y2

    def _extract_keypoints(self, res) -> Dict[int, Tuple[float, float] | None]:
        kps: Dict[int, Tuple[float, float] | None] = {}
        try:
            if res and res[0].keypoints is not None and len(res[0].keypoints) > 0:
                xy = res[0].keypoints.xy[0].cpu().numpy()
                for i in range(xy.shape[0]):
                    kps[i] = (float(xy[i, 0]), float(xy[i, 1]))
        except Exception:
            pass
        return kps

    def _update_stillness(self, bbox, w, h) -> bool:
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        area = max(1.0, (x2 - x1) * (y2 - y1))
        rel_area = area / float(w * h)

        if rel_area < MIN_FACE_AREA:
            self.last_stable_ts = None
            return False

        self.cx_buf.append(cx)
        self.cy_buf.append(cy)
        self.area_buf.append(rel_area)

        need_len = int(STILLNESS_SEC * FPS_ASSUMED * 0.6)
        if len(self.cx_buf) < max(3, need_len):
            self.last_stable_ts = None
            return False

        stdx = float(np.std(self.cx_buf))
        stdy = float(np.std(self.cy_buf))
        stda = float(np.std(self.area_buf))
        stable_now = (
            stdx < MOTION_EPS_CXCY
            and stdy < MOTION_EPS_CXCY
            and stda < MOTION_EPS_AREA
        )

        now = time.time()
        if stable_now:
            if self.last_stable_ts is None:
                self.last_stable_ts = now
            return (now - self.last_stable_ts) >= STILLNESS_SEC
        else:
            self.last_stable_ts = None
            return False

    def _stamp_capture(self):
        """Store current frame & metrics as a capture."""
        self.captured_img = Image.fromarray(self.latest_rgb)
        self.captured_ts = time.time()
        self.captured_metrics = list(self.last_metrics_lines)
        self.last_stable_ts = None

    def capture_now(self) -> bool:
        if self.latest_rgb is None:
            return False
        with self.lock:
            self._stamp_capture()
        return True

    def recv(self, frame):
        import av

        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = img_bgr[:, :, ::-1]
        h, w, _ = img_rgb.shape
        self.latest_rgb = img_rgb

        if self.model:
            res = self.model.predict(img_rgb, imgsz=640, device="cpu", verbose=False)
            plotted = res[0].plot()[:, :, ::-1]  # YOLO provided blue box + green skeleton
            kps = self._extract_keypoints(res)
            lines = format_metrics(kps)
            with self.lock:
                self.last_metrics_lines = lines
            pil = Image.fromarray(plotted)
            pil = draw_tiny_metrics_top_right(pil, lines, size=16, margin=10)
            out_rgb = np.array(pil)
        else:
            with self.lock:
                self.last_metrics_lines = ["(pose model not available)"]
            out_rgb = img_rgb

        bbox = self._detect_bbox(img_rgb) if self.model else None
        if bbox is not None:
            stable = self._update_stillness(bbox, w, h)
            if stable:
                with self.lock:
                    self._stamp_capture()

        return av.VideoFrame.from_ndarray(out_rgb, format="rgb24")


# =================== Streamlit app ===================

st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
    <style>
      .stApp [data-testid=stSidebar] {display: none;}
      .block-container {
        padding-top: 1.2rem;
        padding-bottom: 0.5rem;
        max-width: 1600px;
      }
      .left-col, .right-col {
        height: calc(100vh - 140px);
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
      }
      .left-col video, .left-col canvas {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
      }
      .right-col img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
      }
      .art-wrap {
        flex: 1;
        display: flex;
        flex-direction: column;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"<h1 style='font-family:system-ui;margin-bottom:0.2rem;'>{APP_TITLE}</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='opacity:0.75;margin-bottom:1.2rem;'>"
    "Hold still for ~3–5 seconds to auto-capture, or press the button to capture on demand. "
    "Left: live video with pose. Right: single matched artwork with tiny pink metrics."
    "</p>",
    unsafe_allow_html=True,
)

API_URL = DEFAULT_API_URL

if "last_ts" not in st.session_state:
    st.session_state["last_ts"] = 0.0
if "last_match" not in st.session_state:
    st.session_state["last_match"] = None
if "last_metrics" not in st.session_state:
    st.session_state["last_metrics"] = []
if "countdown_target" not in st.session_state:
    st.session_state["countdown_target"] = None


def force_rerun():
    st.experimental_rerun()


left, right = st.columns(2, gap="large")

with left:
    st.subheader("Live")
    st.markdown('<div class="left-col">', unsafe_allow_html=True)

    ctx = webrtc_streamer(
        key="ear-webrtc",
        mode=WebRtcMode.LIVE,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=CuratorialProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📸 Capture (wait 3s)", use_container_width=True):
            st.session_state["countdown_target"] = time.time() + 3.0
    with col_btn2:
        if st.button("⚡ Instant Capture", use_container_width=True):
            if ctx and ctx.video_processor:
                ok = ctx.video_processor.capture_now()
                st.toast("Captured." if ok else "No frame yet.", icon="✅" if ok else "⚠️")
                if ok:
                    force_rerun()

    if st.session_state["countdown_target"]:
        remain = st.session_state["countdown_target"] - time.time()
        if remain > 0:
            st.info(f"Capturing in {remain:.1f}s… Please hold still.")
        else:
            if ctx and ctx.video_processor:
                ok = ctx.video_processor.capture_now()
                st.toast("Captured." if ok else "No frame yet.", icon="✅" if ok else "⚠️")
            st.session_state["countdown_target"] = None
            force_rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # /left-col

with right:
    st.subheader("Matched artwork")
    st.markdown('<div class="right-col">', unsafe_allow_html=True)
    st.markdown('<div class="art-wrap">', unsafe_allow_html=True)

    ph = st.empty()
    proc: Optional[CuratorialProcessor] = None
    if ctx and ctx.video_processor:
        proc = ctx.video_processor

    if not proc:
        ph.info("Initializing camera…")
    else:
        with proc.lock:
            cap_img = proc.captured_img
            cap_ts = getattr(proc, "captured_ts", 0.0)
            cap_metrics = list(getattr(proc, "captured_metrics", []))

        if cap_img is not None and cap_ts > st.session_state["last_ts"]:
            st.session_state["last_ts"] = cap_ts
            st.session_state["last_metrics"] = cap_metrics

            buf = io.BytesIO()
            cap_img.save(buf, format="JPEG")
            buf.seek(0)

            files = {"image": ("frame.jpg", buf.getvalue(), "image/jpeg")}
            data = {"museum": "mixed", "topk": 3}

            try:
                resp = requests.post(API_URL, files=files, data=data, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
                st.session_state["last_match"] = payload
            except Exception as exc:
                st.session_state["last_match"] = {"error": str(exc)}

            force_rerun()

        payload = st.session_state.get("last_match")

        if not payload:
            ph.info("Hold still or press capture to trigger matching…")
        elif "error" in payload:
            ph.error(f"Error from backend: {payload['error']}")
        else:
            results = payload.get("results") or []
            if not results:
                ph.warning("No matches returned from backend.")
            else:
                # Top-1 only for clean exhibition effect
                top = results[0]
                filename = top.get("filename") or top.get("file")

                img_path = ensure_image_path(filename or "")
                if not img_path:
                    ph.error(f"Image file not found for: {filename}")
                else:
                    painting = safe_open_image(img_path)
                    if painting is None:
                        ph.error(f"Failed to open image: {img_path}")
                    else:
                        meta_row = lookup_meta(str(filename))
                        meta = {
                            "artist": meta_row.get("artist") or top.get("artist") or "artist name",
                            "year": meta_row.get("year") or top.get("year") or "",
                            "price_text": meta_row.get("price_text")
                            or meta_row.get("auction_price_usd")
                            or "",
                        }

                        painted = overlay_right_labels(painting, meta)
                        metrics = st.session_state.get("last_metrics") or []
                        painted = draw_tiny_metrics_top_right(painted, metrics, size=16, margin=12)

                        w = min(RIGHT_IMG_MAXW, painted.width)
                        caption = f"{meta_row.get('title','')} — {meta.get('artist','')}"
                        ph.image(painted, caption=caption, width=w)

    st.markdown("</div>", unsafe_allow_html=True)  # /art-wrap
    st.markdown("</div>", unsafe_allow_html=True)  # /right-col
