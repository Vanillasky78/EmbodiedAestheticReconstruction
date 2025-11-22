from __future__ import annotations

import io
import os
import time
import threading
import random
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

# Use GLOBAL MIXED INDEX (local + met + aic merged)
DATA_DIR = ROOT_DIR / "data" / "mixed"
IMAGES_DIR = DATA_DIR / "images"

# All museum image roots (search order)
MUSEUM_IMAGE_DIRS = [
    DATA_DIR / "images",                    # mixed
    ROOT_DIR / "data" / "local" / "images",
    ROOT_DIR / "data" / "met" / "images",
    ROOT_DIR / "data" / "aic" / "images",
]

# Meta CSV candidates (fallback to local if needed)
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

# Stillness / motion parameters
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

RIGHT_IMG_MAXW = 900

# Unified artwork canvas (让右侧画面比例统一，更接近摄像头画面)
CANVAS_W = 1280
CANVAS_H = 720
CANVAS_RATIO = CANVAS_W / CANVAS_H

# Diversity / randomness control
MAX_RECENT = 6          # how many artworks to remember
PRIMARY_TOP_K = 3       # "very best" range
TAIL_TOP_K = 10         # search in top-10 in total
TAIL_RANDOM_PROB = 0.3  # 30% chance to look into 4–10 first


# =================== UTILITIES ===================

_META_CACHE: Optional[Dict[str, Dict]] = None


def load_meta_mapping() -> Dict[str, Dict]:
    """Load CSV → filename → metadata mapping."""
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
    """
    Resolve an artwork filename to a real image path.

    Strategy:
      1) If it's an absolute existing path → return.
      2) If it has subdirectories, try relative to project root / data.
      3) Otherwise, search by basename in all known museum image dirs.
    """
    if not filename:
        return None

    p = Path(filename)

    # 1) absolute path already
    if p.is_absolute() and p.exists():
        return p

    candidates: List[Path] = []

    # 2) has subdirectory component (e.g. "aic/images/aic_4939.jpg")
    if p.parent != Path("."):
        candidates.append(ROOT_DIR / p)
        candidates.append(ROOT_DIR / "data" / p)

    # 3) search by basename across all museum image roots
    basename = p.name
    for root in MUSEUM_IMAGE_DIRS:
        candidates.append(root / basename)

    for c in candidates:
        if c.exists():
            return c

    return None


def _load_font(size: int = 40):
    """Try Courier → Arial → default."""
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
    """Small pink pose metrics in top-right corner of the image."""
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
    cosv = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def format_metrics(kps: Dict[int, Tuple[float, float] | None]) -> List[str]:
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


def _format_price(meta: Dict) -> str:
    """
    Return a human-readable price string.

    Priority:
      1) Use existing price_text / auction_price_usd / price if present.
      2) Otherwise, generate a *large* deterministic pseudo-price from
         year + filename, just for exhibition visuals (NOT real value).
    """
    raw = (
        meta.get("price_text")
        or meta.get("auction_price_usd")
        or meta.get("price")
    )

    if raw is not None:
        s = str(raw).strip()
        if s and s not in {"—", "-", "None", "nan"}:
            return s

    # ---------- No real price in metadata → fabricate a big pseudo price ----------
    # Year influence
    year = None
    try:
        if meta.get("year"):
            year = int(float(meta["year"]))
    except Exception:
        year = None

    # Big base so prices all look like masterpieces
    base = 2_000_000  # $2M baseline
    if year:
        # older → more expensive
        base += max(0, year - 1400) * 5_000

    # Filename hash for deterministic jitter
    fname = (
        meta.get("filename")
        or meta.get("image_path")
        or meta.get("path")
        or ""
    )
    h = sum(ord(c) for c in str(fname)) % 20_000
    price_val = base + h * 500  # add up to +10M

    # Format into M / k
    if price_val >= 100_000_000:
        return f"${price_val / 1_000_000:.1f}M"
    elif price_val >= 1_000_000:
        return f"${price_val / 1_000_000:.2f}M"
    else:
        return f"${price_val / 1_000:.1f}k"


def overlay_right_labels(painting: Image.Image, meta: Dict) -> Image.Image:
    """Yellow label stack: price, year, artist."""
    im = painting.convert("RGB").copy()
    draw = ImageDraw.Draw(im)

    font_big = _load_font(44)
    font_small = _load_font(36)

    price = _format_price(meta)
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


def format_painting_canvas(painting: Image.Image) -> Image.Image:
    """
    仅做视觉层处理：
    1) 裁剪为统一宽屏比例 (CANVAS_W : CANVAS_H)，让左右画面更对齐
    2) 缩放到统一分辨率
    3) 加一个柔和暗角遮罩，保留中间人物更亮，边缘稍微压暗
    """
    img = painting.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    current_ratio = w / h
    target_ratio = CANVAS_RATIO

    # 居中裁剪到目标比例
    if current_ratio > target_ratio:
        # 太宽 → 裁左右
        new_w = int(h * target_ratio)
        x1 = max(0, (w - new_w) // 2)
        x2 = x1 + new_w
        crop_box = (x1, 0, x2, h)
    else:
        # 太高 → 裁上下
        new_h = int(w / target_ratio)
        y1 = max(0, (h - new_h) // 2)
        y2 = y1 + new_h
        crop_box = (0, y1, w, y2)

    img = img.crop(crop_box)
    img = img.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)

    # 生成暗角遮罩
    yy, xx = np.mgrid[0:CANVAS_H, 0:CANVAS_W]
    cx, cy = CANVAS_W / 2.0, CANVAS_H / 2.0
    nx = (xx - cx) / cx
    ny = (yy - cy) / cy
    r = np.sqrt(nx * nx + ny * ny)

    # 半径 0~1 之间中心亮、边缘暗
    # 0.25 以内几乎不衰减，0.25~1 之间逐渐降到 0.55
    softness = 0.55
    vignette = 1.0 - np.clip((r - 0.25) / (1.0 - 0.25), 0.0, 1.0) * softness
    vignette = vignette[..., None]

    arr = np.asarray(img).astype("float32") / 255.0
    arr = arr * vignette
    arr = np.clip(arr * 255.0, 0, 255).astype("uint8")

    return Image.fromarray(arr)


def force_rerun():
    """Compatible with new/old Streamlit."""
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
    YOLOv8-Pose:
    - blue bbox
    - green skeleton
    - pink tiny pose metrics
    - stillness-based auto capture
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
                self.last_metrics_lines = ["(pose model unavailable)"]
            out_rgb = img_rgb

        bbox = self._detect_bbox(img_rgb) if self.model else None
        if bbox and self._update_stillness(bbox, w, h):
            with self.lock:
                if time.time() - self.captured_ts > 0.35:
                    self._stamp_capture()

        out_bgr = out_rgb[:, :, ::-1]
        return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")


# =================== PAGE LAYOUT ===================

st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    f"""
<style>
section[data-testid="stSidebar"] {{ display: none !important; }}
header, footer, [data-testid="stToolbar"] {{ visibility: hidden !important; }}

body {{
  background-color: #f5f5f7;
}}
.block-container {{
  padding-top: 0.6rem;
  padding-bottom: 0.6rem;
  max-width: 1700px;
}}

h1 {{
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif;
  letter-spacing: 0.04em;
  font-weight: 700;
}}

.left-col .cam-wrap {{
  position: relative;
  height: 80vh;
  width: 100%;
  overflow: hidden;
  border-radius: 18px;
  background: #111;
  box-shadow: 0 18px 45px rgba(0,0,0,0.3);
}}
.left-col .cam-wrap video {{
  height: 100% !important;
  width: auto !important;
  object-fit: cover !important;
  border-radius: 18px !important;
}}

.right-col .art-wrap {{
  position: relative;
  height: 80vh;
  max-width: {RIGHT_IMG_MAXW}px;
  overflow: hidden;
  margin: 0 auto;
  border-radius: 18px;
  background: #050505;
  box-shadow: 0 18px 45px rgba(0,0,0,0.35);
}}
.right-col .art-wrap img {{
  display: block;
  width: 100% !important;
  height: 100% !important;
  object-fit: cover !important;
}}

button[kind="secondary"] {{
  border-radius: 999px !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption(
    "Hold still for ~3–5 seconds to auto-capture, or press the button to capture manually. "
    "Left: live video with pose. Right: matched artwork with tiny pink pose metrics."
)

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
if "recent_files" not in st.session_state:
    st.session_state["recent_files"] = []  # type: List[str]

API_URL = DEFAULT_API_URL


# =================== LEFT PANEL ===================

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

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        if st.button("📸 Capture (wait 3s)", use_container_width=True):
            st.session_state["countdown_target"] = time.time() + 3.0

    with c2:
        if st.button("⚡ Instant Capture", use_container_width=True):
            if ctx and ctx.video_processor:
                ok = ctx.video_processor.capture_now()
                st.toast("Captured." if ok else "Try again.", icon="✅" if ok else "⚠️")

    if st.session_state["countdown_target"]:
        remain = st.session_state["countdown_target"] - time.time()
        if remain > 0:
            st.info(f"Capturing in {remain:.1f}s… Hold still.")
        else:
            if ctx and ctx.video_processor:
                ok = ctx.video_processor.capture_now()
                st.toast("Captured." if ok else "Try again.", icon="✅" if ok else "⚠️")
            st.session_state["countdown_target"] = None
            force_rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =================== RIGHT PANEL ===================

with right:
    st.subheader("Matched artwork")

    # Match mode control
    match_mode = st.selectbox(
        "Match mode",
        ["default", "portrait_only", "high_value_only", "portrait_high_value"],
        index=0,
        help="Use 'portrait_only' to bias towards clear portraits; "
             "'high_value_only' for masterpieces.",
    )

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
            # Ask backend for Top-10, we'll do diversity selection here
            data = {
                "museum": "mixed",
                "topk": TAIL_TOP_K,
                "mode": match_mode,
            }

            try:
                resp = requests.post(API_URL, files=files, data=data, timeout=30)
                resp.raise_for_status()
                st.session_state["last_match"] = resp.json()
            except Exception as exc:
                st.session_state["last_match"] = {"error": str(exc)}

            force_rerun()

        payload = st.session_state.get("last_match")

        if not payload:
            ph.info("Hold still or press capture to match…")
        elif "error" in payload:
            ph.error(f"Backend error: {payload['error']}")
        else:
            results = payload.get("results") or []
            if not results:
                ph.warning("No matches returned.")
            else:
                recent = st.session_state.get("recent_files", [])

                # ---------- diversity + mild randomness ----------
                # Decide whether to start looking from the tail (ranks 4–10)
                use_tail_first = (
                    random.random() < TAIL_RANDOM_PROB
                    and len(results) > PRIMARY_TOP_K
                )
                primary_slice = results[:PRIMARY_TOP_K]
                tail_slice = results[PRIMARY_TOP_K:TAIL_TOP_K]

                ordered_candidates = (
                    (tail_slice + primary_slice)
                    if use_tail_first
                    else (primary_slice + tail_slice)
                )

                chosen = None
                chosen_fname = None

                for r in ordered_candidates:
                    fname = (
                        r.get("filename")
                        or r.get("file")
                        or r.get("image_path")
                        or r.get("path")
                    )
                    if not fname:
                        continue
                    if fname not in recent:
                        chosen = r
                        chosen_fname = fname
                        break

                # Fallback: if all candidates are "recent", use strict Top-1
                if chosen is None:
                    chosen = results[0]
                    chosen_fname = (
                        chosen.get("filename")
                        or chosen.get("file")
                        or chosen.get("image_path")
                        or chosen.get("path")
                    )

                # Update recent memory
                if chosen_fname:
                    if chosen_fname in recent:
                        recent.remove(chosen_fname)
                    recent.insert(0, chosen_fname)
                    del recent[MAX_RECENT:]
                    st.session_state["recent_files"] = recent

                filename = chosen_fname

                img_path = ensure_image_path(filename or "")
                if not img_path:
                    ph.error(f"Image not found: {filename}")
                else:
                    painting = safe_open_image(img_path)
                    if painting is None:
                        ph.error(f"Failed to open: {img_path}")
                    else:
                        # 先统一画布比例 + 暗角遮罩（只影响视觉）
                        painting_canvas = format_painting_canvas(painting)

                        meta_row = lookup_meta(str(filename))
                        meta = {
                            "artist": meta_row.get("artist")
                            or chosen.get("artist")
                            or "artist name",
                            "year": meta_row.get("year") or chosen.get("year") or "",
                            "price_text": meta_row.get("price_text")
                            or meta_row.get("auction_price_usd")
                            or "",
                            "filename": filename,
                        }

                        painted = overlay_right_labels(painting_canvas, meta)
                        metrics = st.session_state.get("last_metrics") or []
                        painted = draw_tiny_metrics_top_right(
                            painted, metrics, size=16, margin=12
                        )

                        w = min(RIGHT_IMG_MAXW, painted.width)
                        caption = f"{meta_row.get('title','')} — {meta.get('artist','')}"
                        ph.image(painted, caption=caption, width=w)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
