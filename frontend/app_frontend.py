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

# =================== PATHS & CONSTANTS ===================

FRONTEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONTEND_DIR.parent

# Use global mixed data (local + met)
DATA_DIR = ROOT_DIR / "data" / "mixed"
IMAGES_DIR = DATA_DIR / "images"

# Meta CSVs: prefer mixed embeddings_meta, then fall back to local CSVs
META_CSV_CANDIDATES = [
    DATA_DIR / "embeddings_meta.csv",
    ROOT_DIR / "data" / "local" / "portrait_works_enhanced_english.csv",
    ROOT_DIR / "data" / "local" / "portrait_works.csv",
]

# Backend API
DEFAULT_API_URL = "http://127.0.0.1:8000/match"

APP_TITLE = "Embodied Aesthetic Reconstruction"

# YOLOv8-Pose model path (placed in frontend/)
YOLO_MODEL_PATH = FRONTEND_DIR / "yolov8n-pose.pt"

# WebRTC config (Safari needs STUN)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Stillness detection
STILLNESS_SEC = 3.5
MAX_BUF_SEC = 5.0
FPS_ASSUMED = 12
MOTION_EPS_CXCY = 4.0
MOTION_EPS_AREA = 0.03
MIN_FACE_AREA = 0.06

# Colors
YELLOW = (255, 235, 59)
BLACK = (0, 0, 0)
HOT_PINK = (255, 30, 180)

# Right image max display width
RIGHT_IMG_MAXW = 900


# =================== META & IMAGE HELPERS ===================

_META_CACHE: Optional[Dict[str, Dict]] = None


def load_meta_mapping() -> Dict[str, Dict]:
    """Load CSV metadata and index by filename."""
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
    """Resolve filename inside IMAGES_DIR."""
    if not filename:
        return None
    p = Path(filename)
    if not p.is_absolute():
        p = IMAGES_DIR / p
    return p if p.exists() else None


# =================== DRAWING HELPERS ===================


def _load_font(size: int = 40) -> ImageFont.FreeTypeFont:
    """Prefer Courier / Courier New, fall back to Arial / default."""
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
    """Draw tiny pink pose metrics in the top-right corner."""
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


def _angle_deg(p1, p2):
    if p1 is None or p2 is None:
        return None
    vx, vy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(np.arctan2(vy, vx)))


def _elbow_angle(shoulder, elbow, wrist):
    if None in (shoulder, elbow, wrist):
        return None
    v1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]], float)
    v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]], float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-5 or n2 < 1e-5:
        return None
    cosv = np.clip(np.dot(v1, n2 and v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def format_metrics(kps: Dict[int, Tuple[float, float] | None]) -> List[str]:
    """Convert keypoints into several formatted lines."""
    le, re = kps.get(1), kps.get(2)
    lsh, rsh = kps.get(5), kps.get(6)
    lel, rel = kps.get(7), kps.get(8)
    lwr, rwr = kps.get(9), kps.get(10)

    fdeg = lambda v: "—" if v is None else f"{v:+.1f}°"
    fpt = lambda p: "(0, 0)" if p is None else f"({int(p[0])}, {int(p[1])})"

    return [
        f"Head tilt: {fdeg(_angle_deg(re, le))}",
        f"Shoulder:  {fdeg(_angle_deg(rsh, lsh))}",
        f"L elbow:   {fdeg(_elbow_angle(lsh, lel, lwr))}",
        f"R elbow:   {fdeg(_elbow_angle(rsh, rel, rwr))}",
        f"L wrist:   {fpt(lwr)}",
        f"R wrist:   {fpt(rwr)}",
    ]


def overlay_right_labels(painting: Image.Image, meta: Dict) -> Image.Image:
    """
    Overlay 3 yellow rectangular labels on the artwork:
      1. price (price_text / auction_price_usd)
      2. year
      3. artist
    Font: Courier, black text on yellow background.
    """
    im = painting.convert("RGB").copy()
    draw = ImageDraw.Draw(im)

    font_big = _load_font(44)
    font_small = _load_font(36)

    price = (
        meta.get("price_text")
        or meta.get("auction_price_usd")
        or meta.get("price")
        or "—"
    )
    year = str(meta.get("year") or "—")
    artist = meta.get("artist") or "artist name"

    lines = [price, year, artist]
    fonts = [font_big, font_small, font_big]

    margin_x = 24
    y = int(im.height * 0.50)

    for text, font in zip(lines, fonts):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        w, h = r - l, b - t

        pad_x, pad_y = 16, 10
        box_w, box_h = w + 2 * pad_x, h + 2 * pad_y

        x = margin_x
        draw.rectangle([x, y, x + box_w, y + box_h], fill=YELLOW)
        draw.text((x + pad_x, y + pad_y), text, fill=BLACK, font=font)

        y += box_h + 10

    return im


def force_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


# =================== YOLO VIDEO PROCESSOR ===================

try:
    from ultralytics import YOLO

    HAS_YOLO = True
except Exception:
    HAS_YOLO = False


class CuratorialProcessor(VideoProcessorBase):
    """
    YOLOv8-Pose overlay + stillness detection + capture.
    """

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
        """Save current frame and metrics as a capture."""
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
            plotted = res[0].plot()[:, :, ::-1]
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
        if bbox and self._update_stillness(bbox, w, h):
            with self.lock:
                if time.time() - self.captured_ts > 0.35:
                    self._stamp_capture()

        out_bgr = out_rgb[:, :, ::-1]
        return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")


# =================== STREAMLIT LAYOUT ===================

st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    f"""
<style>
section[data-testid="stSidebar"] {{ display: none !important; }}
header, footer, [data-testid="stToolbar"] {{ visibility: hidden !important; }}
.block-container {{ padding-top: 0.6rem; padding-bottom: 0.6rem; max-width: 1700px; }}

.left-col .cam-wrap {{
  position: relative;
  height: 92vh;
  width: 100%;
  overflow: hidden;
  border-radius: 12px;
  background: #111;
}}
.left-col .cam-wrap video {{
  height: 100% !important;
  width: auto !important;
  object-fit: cover !important;
  border-radius: 12px !important;
}}

.right-col .art-wrap {{
  position: relative;
  height: 92vh;
  max-width: {RIGHT_IMG_MAXW}px;
  overflow: hidden;
  margin: 0 auto;
}}
.right-col .art-wrap img {{
  display: block;
  width: 100% !important;
  height: 100% !important;
  object-fit: cover !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption(
    "Hold still for ~3–5 seconds to auto-capture, or press the button to capture on demand. "
    "Left: live with pose. Right: single matched artwork with tiny pink metrics."
)

# Slight auto-refresh so right side updates after capture
try:
    st.autorefresh(interval=700, key="ear_auto", limit=None)
except Exception:
    pass

left, right = st.columns([1, 1], gap="large")

if "countdown_target" not in st.session_state:
    st.session_state["countdown_target"] = None
if "last_match" not in st.session_state:
    st.session_state["last_match"] = None
if "last_metrics" not in st.session_state:
    st.session_state["last_metrics"] = []
if "last_ts" not in st.session_state:
    st.session_state["last_ts"] = 0.0

API_URL = DEFAULT_API_URL

# ------------------- LEFT: LIVE CAMERA -------------------

with left:
    st.subheader("Live")
    st.markdown('<div class="left-col">', unsafe_allow_html=True)
    st.markdown('<div class="cam-wrap">', unsafe_allow_html=True)

    ctx = webrtc_streamer(
        key="ear-curatorial-final",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=CuratorialProcessor,
        async_processing=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)  # /cam-wrap
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📸 Capture (wait 3s)", use_container_width=True):
            st.session_state["countdown_target"] = time.time() + 3.0
    with c2:
        if st.button("⚡ Instant Capture", use_container_width=True):
            if ctx and ctx.video_processor:
                ok = ctx.video_processor.capture_now()
                st.toast("Captured." if ok else "No frame yet, try again.", icon="✅" if ok else "⚠️")

    # Countdown logic
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

# ------------------- RIGHT: MATCHED ARTWORK (TOP-1) -------------------

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
        # Read latest capture from processor
        with proc.lock:
            cap_img = proc.captured_img
            cap_ts = getattr(proc, "captured_ts", 0.0)
            cap_metrics = list(getattr(proc, "captured_metrics", []))

        # New capture → send to backend
        if cap_img is not None and cap_ts > st.session_state["last_ts"]:
            st.session_state["last_ts"] = cap_ts
            st.session_state["last_metrics"] = cap_metrics

            buf = io.BytesIO()
            cap_img.save(buf, format="JPEG")
            buf.seek(0)

            files = {"image": ("frame.jpg", buf.getvalue(), "image/jpeg")}
            # Global mixed index, only Top-1
            data = {"museum": "mixed", "topk": 1}

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
                metrics = st.session_state.get("last_metrics") or []
                top = results[0]

                filename = top.get("filename") or top.get("file")
                if not filename:
                    ph.error("Backend did not return a filename.")
                else:
                    img_path = ensure_image_path(filename)
                    if not img_path:
                        ph.error(f"Image file not found for: {filename}")
                    else:
                        painting = safe_open_image(img_path)
                        if painting is None:
                            ph.error(f"Failed to open image: {img_path}")
                        else:
                            meta_row = lookup_meta(str(filename))
                            meta = {
                                "artist": meta_row.get("artist")
                                or top.get("artist")
                                or "artist name",
                                "year": meta_row.get("year")
                                or top.get("year")
                                or "",
                                "price_text": meta_row.get("price_text")
                                or meta_row.get("auction_price_usd")
                                or "",
                            }

                            painted = overlay_right_labels(painting, meta)
                            painted = draw_tiny_metrics_top_right(
                                painted, metrics, size=16, margin=12
                            )

                            w = min(RIGHT_IMG_MAXW, painted.width)
                            title = meta_row.get("title", "") or top.get("title", "")
                            caption = f"{title} — {meta.get('artist','')}"
                            ph.image(painted, caption=caption, width=w)

    st.markdown("</div>", unsafe_allow_html=True)  # /art-wrap
    st.markdown("</div>", unsafe_allow_html=True)  # /right-col
