import os
import io
import cv2
import time
import base64
import zipfile
import numpy as np
from PIL import Image
import streamlit as st

from processors.pose import extract_pose, pose_similarity
from processors.retrieval import load_clip, embed_image, CorpusIndex, fuse_scores

# -----------------------------
# Streamlit 页面样式与布局
# -----------------------------
st.set_page_config(page_title="Embodied Aesthetic Reconstruction", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 0.8rem; }
.card { border-radius: 16px; padding: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); margin-bottom: 1rem; }
img { border-radius: 12px; }
.caption { font-size: 12px; opacity: .65; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 工具函数：调色 / 骨架叠绘 / 背景替换（Poisson）
# -----------------------------
def reinhard_color_transfer(src_bgr: np.ndarray, ref_bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """将 src 的颜色统计匹配到 ref，并按 strength 与原图混合"""
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    for i in range(3):
        s_mean, s_std = src_lab[:, :, i].mean(), src_lab[:, :, i].std() + 1e-6
        r_mean, r_std = ref_lab[:, :, i].mean(), ref_lab[:, :, i].std() + 1e-6
        src_lab[:, :, i] = (src_lab[:, :, i] - s_mean) * (r_std / s_std) + r_mean
    out = np.clip(src_lab, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(src_bgr.astype(np.uint8), 1 - strength, out_bgr, strength, 0)

# 简单骨架点可视化（可换成更精美的骨架连线）
def overlay_pose(image_bgr: np.ndarray, keypoints_norm: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()
    # 默认 keypoints_norm 是 [0,1] 的 bbox 归一化；这里仅绘制相对分布（简化版）
    # 如果你在 app 中只对输入图绘制，可直接把 extract_pose 返回的 bbox 反归一化。
    # 为简单起见，这里把点映射到整幅图中心区域（可替换为更严格的反归一化）
    for x, y in keypoints_norm:
        cx, cy = int(x * w), int(y * h)
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(out, (cx, cy), 3, color, -1)
    return out

def background_replace_poisson(inp_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """MediaPipe 抠图 + OpenCV Poisson 融合，把人物放到艺术品背景里"""
    import mediapipe as mp
    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    mask = mp_selfie.process(cv2.cvtColor(inp_bgr, cv2.COLOR_BGR2RGB)).segmentation_mask
    mask = (mask > 0.5).astype(np.uint8) * 255

    ref_resized = cv2.resize(ref_bgr, (inp_bgr.shape[1], inp_bgr.shape[0]))
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return ref_resized
    center_y, center_x = np.mean(coords, axis=0).astype(int)
    center = (int(center_x), int(center_y))

    mixed = cv2.seamlessClone(
        inp_bgr,           # foreground
        ref_resized,       # background
        mask,              # mask
        center,            # center
        cv2.MIXED_CLONE
    )
    return mixed

# -----------------------------
# YOLO 人体检测 + 智能裁切（可选）
# -----------------------------
_ULTRA_OK = True
try:
    from ultralytics import YOLO
    _yolo_model = YOLO("yolov8n.pt")  # 轻量模型
except Exception:
    _ULTRA_OK = False
    _yolo_model = None

def _enlarge_box(box_xyxy, scale=1.1, img_w=None, img_h=None):
    x1, y1, x2, y2 = map(float, box_xyxy)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    nw, nh = w * scale, h * scale
    nx1, ny1 = cx - nw / 2, cy - nh / 2
    nx2, ny2 = cx + nw / 2, cy + nh / 2
    if img_w is not None and img_h is not None:
        nx1 = max(0, nx1); ny1 = max(0, ny1)
        nx2 = min(img_w, nx2); ny2 = min(img_h, ny2)
    return int(nx1), int(ny1), int(nx2), int(ny2)

def crop_person_with_yolo(pil_img: Image.Image, scale: float = 1.1) -> Image.Image:
    if not _ULTRA_OK or _yolo_model is None:
        return pil_img  # 未安装 ultralytics 时优雅退化
    img_np = np.array(pil_img)  # RGB
    h, w = img_np.shape[:2]
    res = _yolo_model(img_np)[0]
    boxes = []
    for b in res.boxes:
        try:
            if int(b.cls) == 0:  # person
                boxes.append(b.xyxy.cpu().numpy().tolist()[0])
        except Exception:
            continue
    if boxes:
        x1, y1, x2, y2 = _enlarge_box(boxes[0], scale=scale, img_w=w, img_h=h)
        return pil_img.crop((x1, y1, x2, y2))
    return pil_img

# -----------------------------
# 下载链接
# -----------------------------
def _img_to_png_bytes(bgr_img: np.ndarray) -> bytes:
    pil_img = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def get_image_download_link(bgr_img: np.ndarray, filename="image.png") -> str:
    b = _img_to_png_bytes(bgr_img)
    b64 = base64.b64encode(b).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">⬇️ 下载 {filename}</a>'

def get_zip_download_link(images: dict, zip_filename="fusion_results.zip") -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, img in images.items():
            z.writestr(name, _img_to_png_bytes(img))
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:application/zip;base64,{b64}" download="{zip_filename}">⬇️ 下载全部融合结果 (ZIP)</a>'

# -----------------------------
# 加载模型 & 索引（缓存）
# -----------------------------
@st.cache_resource
def _load_all():
    model, preprocess, device = load_clip()
    # 用绝对路径更稳
    ROOT = os.path.dirname(os.path.abspath(__file__))
    idx_paintings = os.path.join(ROOT, "data", "indices", "paintings")
    indices = {"Paintings": CorpusIndex(idx_paintings)}
    return model, preprocess, device, indices

model, preprocess, device, indices = _load_all()

# -----------------------------
# 页面三栏布局
# -----------------------------
st.title("Embodied Aesthetic Reconstruction — 展演版")

# 控制项（上方）
with st.expander("高级设置（可选）", expanded=False):
    st.write("Index 路径：", list(indices.keys()))
    st.write("YOLO 裁切已启用：" + ("✅" if _ULTRA_OK else "❌（未安装 ultralytics）"))

colL, colM, colR = st.columns([1.15, 1.4, 1.45])

# 左栏：输入 & 控制
with colL:
    st.subheader("1) 输入")
    uploaded = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "webp"])
    crop_mode = st.radio("输入图像模式", ["原图", "自动人物裁切"], horizontal=True)
    corpus = st.selectbox("语料库", list(indices.keys()))
    topk = st.slider("Top-K", 1, 12, 6)
    alpha = st.slider("融合权重 α（视觉 vs 姿态）", 0.0, 1.0, 0.8, 0.05)
    color_strength = st.slider("调色强度", 0.0, 1.0, 0.5, 0.05)
    presentation_mode = st.checkbox("开启展演模式（自动轮播）", value=False)
    fullscreen_mode = st.checkbox("全屏模式（隐藏左/中栏）", value=False)
    play_sec = st.slider("展演切换间隔（秒）", 2, 12, 5)
    run_btn = st.button("Find Matches", type="primary")

# 中栏：Top-K 匹配
results = []
img_for_infer = None
qpose = None

with colM:
    st.subheader("2) 匹配结果")
    if uploaded:
        src_img = Image.open(uploaded).convert("RGB")
        if crop_mode == "自动人物裁切":
            proc_img = crop_person_with_yolo(src_img, scale=1.12)
        else:
            proc_img = src_img
        st.image([src_img, proc_img], caption=["原图", "用于检索的图像"], use_container_width=True)
        img_for_infer = proc_img

    if run_btn and img_for_infer is not None:
        with st.spinner("检索中……"):
            qfeat = embed_image(model, preprocess, device, img_for_infer)
            qpose_res = extract_pose(img_for_infer)
            qpose = qpose_res.keypoints if qpose_res.ok else None

            idx = indices[corpus]
            sims, ids = idx.search(qfeat, max(topk * 3, topk))  # 扩大候选池
            fused_list = []
            for s, i in zip(sims, ids):
                path = idx.paths[i]
                pscore = None
                if qpose is not None and idx.poses is not None:
                    cpose = idx.poses[i]
                    if cpose is not None:
                        pscore = pose_similarity(qpose, cpose)
                final = fuse_scores(float(s), pscore, alpha)
                fused_list.append((path, float(s), pscore, final))

            fused_list.sort(key=lambda x: x[3], reverse=True)
            results = fused_list[:topk]

        if results:
            for path, cs, ps, fs in results:
                with st.container():
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    cap = f"CLIP={cs:.3f}"
                    if ps is not None: cap += f" | Pose={ps:.3f}"
                    cap += f" | Fused={fs:.3f}"
                    st.image(path, caption=cap, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

# 右栏：融合输出
with colR:
    st.subheader("3) 融合结果")
    if results and img_for_infer is not None:
        ref_path = results[0][0]
        ref_bgr = cv2.imread(ref_path)
        inp_bgr = cv2.cvtColor(np.array(img_for_infer), cv2.COLOR_RGB2BGR)

        # 三种融合
        fused_color = reinhard_color_transfer(inp_bgr, ref_bgr, strength=color_strength)
        fused_pose = overlay_pose(inp_bgr.copy(), qpose) if qpose is not None else inp_bgr.copy()
        fused_bg = background_replace_poisson(inp_bgr, ref_bgr)

        if fullscreen_mode:
            st.write("🎬 全屏展演模式")
            if presentation_mode:
                ph = st.empty()
                images = [("🎨 调色融合", fused_color),
                          ("🦴 骨架叠绘", fused_pose),
                          ("🖼️ 背景替换", fused_bg)]
                # 循环一轮即可（Streamlit 不建议死循环阻塞）
                for caption, img_out in images:
                    with ph.container():
                        st.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)
                    time.sleep(play_sec)
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(cv2.cvtColor(fused_color, cv2.COLOR_BGR2RGB), caption="🎨 调色融合", use_container_width=True)
                with c2:
                    st.image(cv2.cvtColor(fused_pose, cv2.COLOR_BGR2RGB), caption="🦴 骨架叠绘", use_container_width=True)
                with c3:
                    st.image(cv2.cvtColor(fused_bg, cv2.COLOR_BGR2RGB), caption="🖼️ 背景替换", use_container_width=True)
        else:
            st.write("🔍 三合一对比")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(cv2.cvtColor(fused_color, cv2.COLOR_BGR2RGB), caption="🎨 调色融合", use_container_width=True)
                st.markdown(get_image_download_link(fused_color, "fusion_color.png"), unsafe_allow_html=True)
            with c2:
                st.image(cv2.cvtColor(fused_pose, cv2.COLOR_BGR2RGB), caption="🦴 骨架叠绘", use_container_width=True)
                st.markdown(get_image_download_link(fused_pose, "fusion_pose.png"), unsafe_allow_html=True)
            with c3:
                st.image(cv2.cvtColor(fused_bg, cv2.COLOR_BGR2RGB), caption="🖼️ 背景替换", use_container_width=True)
                st.markdown(get_image_download_link(fused_bg, "fusion_bg.png"), unsafe_allow_html=True)

            imgs = {"fusion_color.png": fused_color, "fusion_pose.png": fused_pose, "fusion_bg.png": fused_bg}
            st.markdown(get_zip_download_link(imgs), unsafe_allow_html=True)

        st.caption(f"参考作品: {os.path.basename(ref_path)}")
    elif run_btn and img_for_infer is None:
        st.warning("请先上传图片。")
