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

# ----- Paths -----
FRONTEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONTEND_DIR.parent
DATA_DIR = ROOT_DIR / "data"

# default museum key for backend (matches data/local/images etc.)
DEFAULT_MUSEUM = "local"

# YOLOv8 Pose model path (kept in frontend folder)
YOLO_MODEL_PATH = FRONTEND_DIR / "yolov8n-pose.pt"

# WebRTC STUN config (keep default)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Backend API
DEFAULT_API_URL = "http://127.0.0.1:8000/match"


# ===== Helper: font loader =====


def load_courier_like_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Try to load a Courier-style monospaced font.
    Fall back to default PIL font if none found.
    """
    candidate_paths = [
        "Courier New.ttf",
        "cour.ttf",
        "/System/Library/Fonts/Courier.dfont",
        "/Library/Fonts/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for path in candidate_paths:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def overlay_price_year_artist(base_img: Image.Image, meta: Dict) -> Image.Image:
    """
    Draw three yellow labels on top of the artwork image:

    1. price_text / auction_price_usd
    2. year
    3. artist

    All with yellow background, black text, Courier-like font.
    """
    img = base_img.copy().convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # ----- text values -----
    price = meta.get("price_text")
    if not price:
        auction = meta.get("auction_price_usd") or meta.get("price_estimate_usd")
        if auction:
            try:
                val = float(auction)
                price = f"¥{val:,.0f}"
            except Exception:
                price = str(auction)
        else:
            price = "N/A"

    year = meta.get("year") or "Unknown year"
    artist = meta.get("artist") or "Unknown artist"

    price_str = str(price)
    year_str = str(year)
    artist_str = str(artist)

    # ----- fonts & layout -----
    big_size = max(int(h * 0.06), 18)
    mid_size = max(int(h * 0.045), 14)

    font_big = load_courier_like_font(big_size)
    font_mid = load_courier_like_font(mid_size)

    margin_x = int(w * 0.03)
    margin_y = int(h * 0.03)
    pad_x = int(w * 0.015)
    pad_y = int(h * 0.01)
    gap_y = int(h * 0.012)

    def draw_label(y_bottom: int, text: str, font: ImageFont.FreeTypeFont) -> int:
        # measure text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        rect_w = text_w + pad_x * 2
        rect_h = text_h + pad_y * 2

        x0 = margin_x
        y0 = y_bottom - rect_h
        x1 = x0 + rect_w
        y1 = y_bottom

        # yellow rectangle + black text
        draw.rectangle([x0, y0, x1, y1], fill="#ffd800")
        draw.text((x0 + pad_x, y0 + pad_y), text, font=font, fill="black")

        # return next baseline (above this label)
        return y0 - gap_y

    # draw from bottom to top: artist (bottom), year, price (top)
    current_bottom = h - margin_y
    current_bottom = draw_label(current_bottom, artist_str, font_big)
    current_bottom = draw_label(current_bottom, year_str, font_mid)
    draw_label(current_bottom, price_str, font_big)

    return img


# ===== Video Processor (YOLO skeleton) =====


class YoloPoseProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.model = YOLO(str(YOLO_MODEL_PATH))
        self.latest_frame_bgr: Optional[np.ndarray] = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        self.latest_frame_bgr = img_bgr.copy()

        # Run YOLOv8 pose
        results = self.model(img_bgr, verbose=False)
        annotated = results[0].plot() if len(results) > 0 else img_bgr

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ===== Streamlit UI =====


def main() -> None:
    st.set_page_config(
        page_title="EAR — Skeleton Frontend (Final)",
        layout="wide",
    )

    st.title("Embodied Aesthetic Reconstruction — Final Frontend")

    # Sidebar controls
    with st.sidebar:
        st.header("Matching Settings")
        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        museum = st.text_input(
            "Museum key (e.g., local / met)",
            value=DEFAULT_MUSEUM,
        )
        topk = st.number_input(
            "Top-K matches", min_value=1, max_value=10, value=3, step=1
        )
        st.markdown("---")
        st.caption("Left: camera + skeleton overlay. Right: matched artworks via API.")

    col_left, col_right = st.columns(2)

    # ----- Left: Camera with skeleton -----
    with col_left:
        st.subheader("Camera · Skeleton View")

        ctx = webrtc_streamer(
            key="ear-skeleton-final",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=YoloPoseProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

        st.markdown(
            "<small>Tip: if Safari asks for permission, click the red camera icon in the address bar.</small>",
            unsafe_allow_html=True,
        )

        analyze_clicked = st.button("Analyze current frame", type="primary")

    # ----- Right: Results -----
    with col_right:
        st.subheader("Matched Artworks")
        result_placeholder = st.empty()

    # ----- Handle Analyze button -----
    if analyze_clicked:
        if not ctx or not ctx.state.playing:
            result_placeholder.error(
                "Camera is not running. Please start the stream on the left first."
            )
            return

        processor: YoloPoseProcessor = ctx.video_processor  # type: ignore
        if processor is None or processor.latest_frame_bgr is None:
            result_placeholder.error(
                "No frame captured yet. Hold still for a moment and try again."
            )
            return

        # Convert frame to JPEG bytes
        frame_bgr = processor.latest_frame_bgr
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        data = {
            "museum": museum or DEFAULT_MUSEUM,
            "topk": str(int(topk)),
        }
        files = {
            "file": ("frame.jpg", buf, "image/jpeg"),
        }

        try:
            resp = requests.post(api_url, data=data, files=files, timeout=90)
        except Exception as exc:
            result_placeholder.error(f"Request failed: {exc}")
            return

        if resp.status_code != 200:
            result_placeholder.error(
                f"Server returned {resp.status_code}: {resp.text[:300]}"
            )
            return

        try:
            payload = resp.json()
        except json.JSONDecodeError:
            result_placeholder.error("Failed to decode JSON response from API.")
            return

        results: List[Dict] = payload.get("results") or []
        if not results:
            result_placeholder.warning("API returned no matches.")
            return

        # ----- Render results -----
        with result_placeholder.container():
            for idx, r in enumerate(results):
                filename = r.get("filename")
                title = r.get("title") or "(untitled)"
                artist = r.get("artist") or "Unknown artist"
                year = r.get("year") or "Unknown year"
                score = r.get("score")

                st.markdown(f"### #{idx + 1} — {title}")

                # Try to load local image
                img = None
                if filename:
                    candidates = [
                        DATA_DIR / (museum or DEFAULT_MUSEUM) / "images" / filename,
                        DATA_DIR / "local" / "images" / filename,
                        DATA_DIR / "images" / filename,
                    ]
                    for p in candidates:
                        if p.is_file():
                            try:
                                img = Image.open(p).convert("RGB")
                                break
                            except Exception:
                                continue

                if img is not None:
                    img_with_labels = overlay_price_year_artist(img, r)
                    st.image(img_with_labels, use_column_width=True)
                else:
                    st.info(f"(Image file not found for {filename!r})")

                meta_lines = [
                    f"**Artist:** {artist}",
                    f"**Year:** {year}",
                ]
                price = r.get("price_text") or r.get("auction_price_usd")
                if price:
                    meta_lines.append(f"**Price:** {price}")
                if score is not None:
                    try:
                        meta_lines.append(f"**Similarity score:** {float(score):.3f}")
                    except Exception:
                        meta_lines.append(f"**Similarity score:** {score}")

                st.markdown("<br>".join(meta_lines), unsafe_allow_html=True)
                st.markdown("---")


if __name__ == "__main__":
    main()
