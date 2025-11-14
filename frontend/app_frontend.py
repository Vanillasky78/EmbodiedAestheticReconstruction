import io
import json
from pathlib import Path
from typing import List, Dict, Optional

import av
import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

from ultralytics import YOLO

# ==== Paths ====
FRONTEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONTEND_DIR.parent
DATA_DIR = ROOT_DIR / "data"
LOCAL_IMAGES_DIR = DATA_DIR / "local" / "images"
LOCAL_META_CSV = DATA_DIR / "local" / "embeddings_meta.csv"

DEFAULT_MUSEUM = "local"
YOLO_MODEL_PATH = FRONTEND_DIR / "yolov8n-pose.pt"

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)


# ==== Metadata helpers ====

@st.cache_data(show_spinner=False)
def load_metadata(csv_path: Path) -> Dict[str, Dict]:
    """
    Load metadata CSV as a dict keyed by filename.
    """
    import csv

    meta: Dict[str, Dict] = {}
    if not csv_path.exists():
        return meta

    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            meta[filename] = row
    return meta


def format_price(meta_row: Dict) -> Optional[str]:
    """
    优先使用 price_text，如果没有再尝试 auction_price_usd。
    """
    price_text = (meta_row.get("price_text") or "").strip()
    auction_price = (meta_row.get("auction_price_usd") or "").strip()

    if price_text:
        return price_text
    if auction_price:
        # 简单格式化一下，例如 4500000 -> "¥4,500,000"
        try:
            val = float(auction_price)
            return f"¥{val:,.0f}"
        except Exception:
            return auction_price
    return None


def draw_price_year_artist_label(
    img: Image.Image,
    meta_row: Dict,
) -> Image.Image:
    """
    在图像上叠加三条黄色标签（从上到下）：
    1. 价格
    2. 年份
    3. 艺术家
    """

    # 取字段
    price_text = format_price(meta_row) or "N/A"
    year_text = (meta_row.get("year") or "").strip() or "Unknown year"
    artist_text = (meta_row.get("artist") or "").strip() or "Unknown artist"

    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # 字体：优先 Courier / Courier New，失败就用默认
    def _get_font(size: int) -> ImageFont.FreeTypeFont:
        for name in ["Courier New", "Courier"]:
            try:
                return ImageFont.truetype(name, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    font_big = _get_font(size=int(H * 0.045))
    font_mid = _get_font(size=int(H * 0.035))

    # 三行文本
    line1 = price_text
    line2 = str(year_text)
    line3 = artist_text

    # 计算每行文本宽度
    pad_x = int(W * 0.02)
    pad_y = int(H * 0.008)

    w1, h1 = draw.textsize(line1, font=font_big)
    w2, h2 = draw.textsize(line2, font=font_mid)
    w3, h3 = draw.textsize(line3, font=font_big)

    box_width = max(w1, w2, w3) + pad_x * 2

    # 从图像左下角往上堆叠三条
    margin_bottom = int(H * 0.06)
    gap = int(H * 0.008)

    y3_bottom = H - margin_bottom
    y3_top = y3_bottom - h3 - pad_y * 2

    y2_bottom = y3_top - gap
    y2_top = y2_bottom - h2 - pad_y * 2

    y1_bottom = y2_top - gap
    y1_top = y1_bottom - h1 - pad_y * 2

    x_left = int(W * 0.02)
    x_right = x_left + box_width

    # 统一用亮黄色
    yellow = (255, 230, 0, 255)

    # 绘制矩形 + 文本
    def _draw_box(y_top, y_bottom, text, font):
        draw.rectangle([x_left, y_top, x_right, y_bottom], fill=yellow)
        text_w, text_h = draw.textsize(text, font=font)
        text_x = x_left + pad_x
        text_y = y_top + (y_bottom - y_top - text_h) / 2
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0, 255))

    _draw_box(y1_top, y1_bottom, line1, font_big)
    _draw_box(y2_top, y2_bottom, line2, font_mid)
    _draw_box(y3_top, y3_bottom, line3, font_big)

    return img


# ==== YOLO Video Processor ====

class YoloPoseProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[YOLO] = None
        self.latest_frame_bgr: Optional[np.ndarray] = None

    def _ensure_model(self):
        if self.model is None:
            self.model = YOLO(str(YOLO_MODEL_PATH))

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        self.latest_frame_bgr = img_bgr.copy()

        self._ensure_model()
        results = self.model(img_bgr, verbose=False)

        if not results:
            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

        res = results[0]
        annotated = img_bgr.copy()

        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy.tolist()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"person {conf:.2f}",
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        if res.keypoints is not None and res.keypoints.xy is not None:
            kps = res.keypoints.xy.cpu().numpy()  # [num_person, num_kpts, 2]
            for person_kps in kps:
                for x, y in person_kps:
                    cv2.circle(
                        annotated,
                        (int(x), int(y)),
                        4,
                        (0, 255, 0),
                        -1,
                    )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ==== Streamlit Layout ====

st.set_page_config(
    page_title="EAR — Skeleton Frontend (Final)",
    layout="wide",
)

st.markdown(
    "<h1 style='font-size:40px;'>Embodied Aesthetic Reconstruction — Final Frontend</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-size:18px; color:#555;'>"
    "Left: camera view with YOLOv8-Pose skeleton overlay. "
    "Right: matched artworks via API, with price / year / artist labels."
    "</p>",
    unsafe_allow_html=True,
)

# Sidebar: matching settings
with st.sidebar:
    st.header("Matching Settings")

    api_url = st.text_input(
        "API URL",
        value="http://127.0.0.1:8000/match",
        help="Your FastAPI /match endpoint.",
    )

    museum_key = st.text_input(
        "Museum key (e.g., local / met)",
        value=DEFAULT_MUSEUM,
    )

    topk = st.number_input(
        "Top-K matches",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )

    st.markdown(
        "<p style='font-size:12px;color:#777;'>"
        "Tip: If the API is running on another machine, update the URL accordingly."
        "</p>",
        unsafe_allow_html=True,
    )

# Main two-column layout (camera left, artworks right)
col_left, col_right = st.columns(2, gap="large")

# ---- Left: camera & skeleton ----
with col_left:
    st.subheader("Camera · Skeleton View")

    webrtc_ctx = webrtc_streamer(
        key="ear-skeleton-final",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=YoloPoseProcessor,
        async_processing=True,
    )

    analyze_btn = st.button("Analyze current frame", type="primary")

# ---- Right: results ----
with col_right:
    st.subheader("Matched Artworks")

    result_placeholder = st.empty()

    # 处理分析按钮
    if analyze_btn:
        if not webrtc_ctx or not webrtc_ctx.video_processor:
            result_placeholder.error("Camera not ready. Please allow camera access and try again.")
        else:
            processor: YoloPoseProcessor = webrtc_ctx.video_processor
            if processor.latest_frame_bgr is None:
                result_placeholder.error("No frame captured yet. Please wait for camera to start.")
            else:
                frame_bgr = processor.latest_frame_bgr
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=95)
                buf.seek(0)

                data = {
                    "museum": museum_key or DEFAULT_MUSEUM,
                    "topk": str(int(topk)),
                }

                # 关键：后端字段名是 "image"
                files = {
                    "image": ("frame.jpg", buf, "image/jpeg"),
                }

                try:
                    resp = requests.post(api_url, data=data, files=files, timeout=90)
                except Exception as exc:
                    result_placeholder.error(f"Request failed: {exc}")
                else:
                    if resp.status_code != 200:
                        result_placeholder.error(
                            f"Server returned {resp.status_code}: {resp.text}"
                        )
                    else:
                        payload = resp.json()
                        # 安全兜底
                        results: List[Dict] = payload.get("results") or []
                        museum_used = payload.get("museum", museum_key)

                        meta_map = load_metadata(LOCAL_META_CSV)

                        blocks = []
                        for idx, r in enumerate(results):
                            filename = r.get("filename") or ""
                            score = float(r.get("score") or 0.0)

                            # 合并元数据
                            meta = dict(meta_map.get(filename, {}))
                            for k in ["title", "artist", "year", "price_text",
                                      "auction_price_usd", "license", "museum"]:
                                if r.get(k) is not None:
                                    meta[k] = r.get(k)

                            title = meta.get("title") or filename
                            artist = meta.get("artist") or "Unknown artist"
                            year = meta.get("year") or "Unknown year"

                            img_path = LOCAL_IMAGES_DIR / filename

                            # 构建展示块
                            block_html = f"<h4>#{idx+1}: {title}</h4>"
                            block_html += "<ul>"
                            block_html += f"<li><b>Artist:</b> {artist}</li>"
                            block_html += f"<li><b>Year:</b> {year}</li>"
                            block_html += f"<li><b>Score:</b> {score:.3f}</li>"
                            block_html += f"<li><b>Filename:</b> {filename}</li>"
                            block_html += "</ul>"

                            blocks.append((img_path, meta, block_html))

                        # 在 placeholder 中实际渲染
                        with result_placeholder.container():
                            st.markdown(
                                f"<p style='font-size:14px;color:#666;'>"
                                f"Museum key: <b>{museum_used}</b> · Top-{len(blocks)} matches"
                                f"</p>",
                                unsafe_allow_html=True,
                            )

                            for img_path, meta, html in blocks:
                                st.markdown("---")
                                cols_img, cols_meta = st.columns([2, 1])

                                with cols_img:
                                    if img_path.exists():
                                        try:
                                            img = Image.open(img_path).convert("RGB")
                                            img = draw_price_year_artist_label(img, meta)
                                            st.image(
                                                img,
                                                use_column_width=True,
                                                caption=img_path.name,
                                            )
                                        except Exception as exc:
                                            st.error(f"Failed to load image {img_path.name}: {exc}")
                                    else:
                                        st.warning(f"Image not found: {img_path.name}")

                                with cols_meta:
                                    st.markdown(html, unsafe_allow_html=True)

                            if not blocks:
                                st.info("No results returned from API.")


# 一点底部说明
st.markdown(
    "<hr/><p style='font-size:12px;color:#999;'>"
    "EAR Final Frontend — camera + skeleton on the left, CLIP-based artwork matches on the right."
    "</p>",
    unsafe_allow_html=True,
)
