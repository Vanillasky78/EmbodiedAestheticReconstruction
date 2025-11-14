# frontend/app_frontend_final.py
#
# Embodied Aesthetic Reconstruction — Final Frontend
# Layout: two wide columns
#   Left  : live camera + YOLOv8-Pose skeleton overlay
#   Right : matched artworks from backend /match API
#
# Requirements (in your ear-mvp env):
#   pip install streamlit streamlit-webrtc ultralytics pillow numpy requests av opencv-python

import io
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from ultralytics import YOLO

# ---------- Paths ----------
FRONTEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONTEND_DIR.parent
DATA_DIR = ROOT_DIR / "data"

DEFAULT_API_URL = "http://127.0.0.1:8000/match"
DEFAULT_MUSEUM_KEY = "local"
DEFAULT_TOPK = 3

YOLO_MODEL_PATH = FRONTEND_DIR / "yolov8n-pose.pt"

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------- Fonts & drawing helpers ----------


@st.cache_resource(show_spinner=False)
def load_font(size: int = 36) -> ImageFont.FreeTypeFont:
    """
    Try to load a Courier-like monospaced font.
    If not found, fall back to default PIL font.
    """
    candidates = [
        "Courier New.ttf",
        "Courier New Bold.ttf",
        "CourierPrime-Regular.ttf",
        "Menlo.ttc",
        "Monaco.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def overlay_price_year_artist(
    img: Image.Image,
    artist: Optional[str],
    year: Optional[str],
    price_text: Optional[str],
) -> Image.Image:
    """
    Draw three yellow bars on top of the artwork:
      1) price_text
      2) year
      3) artist
    All in Courier-like font, black text.
    """
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Font sizes relative to image height
    font_large = load_font(max(int(h * 0.055), 18))  # price & artist
    font_medium = load_font(max(int(h * 0.045), 16))  # year

    padding_x = int(w * 0.025)
    padding_y = int(h * 0.012)
    gap_y = int(h * 0.012)

    def _draw_bar(text: str, y: int, font: ImageFont.FreeTypeFont) -> int:
        if not text:
            return y
        text = str(text)
        tw, th = draw.textsize(text, font=font)
        box = (
            padding_x,
            y,
            padding_x + tw + padding_x * 2,
            y + th + padding_y * 2,
        )
        draw.rectangle(box, fill=(255, 230, 0))
        text_x = padding_x + padding_x * 0.7
        text_y = y + padding_y * 0.9
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))
        return y + th + padding_y * 2 + gap_y

    # place bars in lower-left quadrant
    current_y = int(h * 0.65)
    current_y = _draw_bar(price_text or "", current_y, font_large)
    current_y = _draw_bar(
        str(year) if year not in (None, "", "None") else "", current_y, font_medium
    )
    _draw_bar(artist or "", current_y, font_large)

    return img


def find_artwork_image(filename: str, museum_key: str) -> Optional[Path]:
    """
    Robustly resolve artwork image path.
    We try several fallbacks to avoid crashes.
    """
    if not filename:
        return None

    museum_key_str = str(museum_key).strip() if museum_key else DEFAULT_MUSEUM_KEY
    candidate_paths = [
        DATA_DIR / museum_key_str / "images" / filename,
        DATA_DIR / "local" / "images" / filename,
        DATA_DIR / "images" / filename,
    ]

    for p in candidate_paths:
        if p.is_file():
            return p
    return None


def format_price(result: Dict) -> Optional[str]:
    """
    Choose a displayable price string:
    1) 'price_text' if present (e.g. '¥4,500,000')
    2) formatted 'auction_price_usd' if numeric
    """
    if result.get("price_text"):
        return str(result["price_text"])

    value = result.get("auction_price_usd")
    if isinstance(value, (int, float)):
        # your dataset uses yuan-like notation, but we keep '¥' as per your design
        return f"¥{value:,.0f}"
    return None


# ---------- YOLOv8 Pose video processor ----------


class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_frame_bgr: Optional[np.ndarray] = None

        # Load YOLO model once (cached in process)
        self.model = YOLO(str(YOLO_MODEL_PATH))

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Save a copy of the raw frame for later "instant capture"
        with self.lock:
            self.latest_frame_bgr = img.copy()

        # Run pose detection
        results = self.model(img, verbose=False)
        annotated = img.copy()

        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue

            # Draw bounding boxes
            if r.boxes.xyxy is not None and len(r.boxes.xyxy) > 0:
                for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
                    x1, y1, x2, y2 = box.int().tolist()
                    cv2.rectangle(
                        annotated, (x1, y1), (x2, y2), (255, 0, 0), 2
                    )  # blue
                    label = f"person {float(conf):.2f}"
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # Draw keypoints skeleton
            if r.keypoints is not None and r.keypoints.xy is not None:
                kps = r.keypoints.xy[0].cpu().numpy().astype(int)
                for x, y in kps:
                    cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame_bgr is None:
                return None
            return self.latest_frame_bgr.copy()


# ---------- API call & result rendering ----------


def call_match_api(
    api_url: str,
    frame_bgr: np.ndarray,
    museum_key: str,
    topk: int,
    timeout: float = 30.0,
) -> Dict:
    """
    Send current frame to backend /match API and return JSON response.
    """
    # Encode frame as JPEG
    success, encoded = cv2.imencode(".jpg", frame_bgr)
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG.")
    img_bytes = encoded.tobytes()

    data = {
        "museum": str(museum_key),
        "topk": str(int(topk)),
    }
    files = {"image": ("frame.jpg", img_bytes, "image/jpeg")}

    resp = requests.post(api_url, data=data, files=files, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def render_results_column(
    col,
    payload: Optional[Dict],
    museum_key: str,
) -> None:
    """
    Render matched artworks (payload from backend) in the right column.
    """
    if not payload:
        col.info("No matching results yet. Capture a frame from the camera to start.")
        return

    results: List[Dict] = payload.get("results", []) or []
    museum = payload.get("museum", museum_key)

    if not results:
        col.warning("Backend returned an empty result list.")
        return

    for rank, r in enumerate(results, start=1):
        filename = r.get("filename") or r.get("file") or ""
        title = r.get("title") or "(Untitled)"
        artist = r.get("artist") or ""
        year = r.get("year")
        score = r.get("score", 0.0)
        price = format_price(r)

        col.markdown("---")
        col.markdown(f"### #{rank}: {title}")

        # Load and render artwork image
        img_path = find_artwork_image(filename, museum)
        if img_path is None:
            col.error(
                f"Failed to load image **{filename}** for museum key **{museum}**."
            )
        else:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                pil_img = overlay_price_year_artist(
                    pil_img, artist=artist, year=year, price_text=price
                )
                col.image(pil_img, use_container_width=True)
            except Exception as exc:
                col.error(
                    f"Error while rendering image **{filename}**: {exc}",
                )

        # Text metadata
        meta_lines = []
        if artist:
            meta_lines.append(f"- **Artist:** {artist}")
        if year not in (None, "", "None"):
            meta_lines.append(f"- **Year:** {year}")
        meta_lines.append(f"- **Score:** {float(score):.3f}")
        if filename:
            meta_lines.append(f"- **Filename:** `{filename}`")

        license_ = r.get("license")
        museum_label = r.get("museum") or museum
        significance = r.get("significance_text")
        if museum_label:
            meta_lines.append(f"- **Collection / Museum:** {museum_label}")
        if license_:
            meta_lines.append(f"- **License:** {license_}")

        if meta_lines:
            col.markdown("\n".join(meta_lines))

        if significance:
            col.markdown(f"> {significance}")


# ---------- Streamlit main ----------


def main():
    st.set_page_config(
        page_title="EAR — Final Frontend",
        layout="wide",
    )

    st.markdown(
        "<h1 style='text-align: center;'>Embodied Aesthetic Reconstruction — Final Frontend</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;'>Left: live camera with YOLOv8-Pose skeleton. "
        "Right: matched artworks from the CLIP-based backend API.</p>",
        unsafe_allow_html=True,
    )

    # --- Settings row (top, not sidebar) ---
    st.markdown("### Matching settings")
    c1, c2, c3 = st.columns([3, 2, 1])

    with c1:
        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    with c2:
        museum_key = st.text_input("Museum key (e.g. local / met)", value=DEFAULT_MUSEUM_KEY)
    with c3:
        topk = st.number_input("Top-K matches", min_value=1, max_value=10, value=DEFAULT_TOPK)

    st.markdown("---")

    # --- Two main columns: left = camera, right = artworks ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Live pose view")

        webrtc_ctx = webrtc_streamer(
            key="ear-final-webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=PoseVideoProcessor,
        )

        st.markdown("")
        capture_btn = st.button("⚡ Instant capture & match", use_container_width=True)
        st.caption(
            "Tip: hold your pose, then click the button to send the current frame to the backend."
        )

    # Use session_state to preserve last payload
    if "last_match_payload" not in st.session_state:
        st.session_state["last_match_payload"] = None

    # When button pressed, call backend
    if capture_btn:
        if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            frame = webrtc_ctx.video_processor.get_latest_frame()
            if frame is None:
                st.warning("No frame captured yet. Please wait a second and try again.")
            else:
                try:
                    with st.spinner("Sending current frame to backend and waiting for matches..."):
                        payload = call_match_api(
                            api_url=api_url,
                            frame_bgr=frame,
                            museum_key=museum_key,
                            topk=int(topk),
                        )
                    st.session_state["last_match_payload"] = payload
                except requests.RequestException as exc:
                    st.error(f"Request to backend failed: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error while calling backend: {exc}")
        else:
            st.warning("Camera is not running yet. Please allow camera access and start the stream.")

    # --- Right column: render matches (using last payload from state) ---
    with col_right:
        st.subheader("Matched artworks")
        render_results_column(
            col_right,
            st.session_state.get("last_match_payload"),
            museum_key=museum_key,
        )

    st.markdown("---")
    st.caption(
        "EAR Final Frontend — left: YOLOv8-Pose skeleton; right: CLIP-based artwork matches "
        "with price/year/artist overlays."
    )


if __name__ == "__main__":
    main()
