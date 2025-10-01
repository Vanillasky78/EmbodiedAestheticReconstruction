"""
Make visual montages: Input image + Top-K matches (with scores) in one canvas.

Usage A (run retrieval now):
    python make_montage.py --folder examples/inputs --index data/indices/paintings --outdir montages --topk 5

Usage B (use existing CSV from batch_test.py):
    python make_montage.py --csv batch_results.csv --outdir montages_csv --topk 5

Notes:
- Works on Windows/Linux/macOS.
- Requires: pillow, opencv-python, numpy, tqdm
  (and for retrieval mode: open_clip_torch, torch, faiss-cpu, mediapipe)
"""

import os
import csv
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ------- Optional: retrieval imports (only needed when --folder mode) -------
RETRIEVAL_AVAILABLE = True
try:
    from processors.retrieval import load_clip, embed_image, CorpusIndex, fuse_scores
    from processors.pose import extract_pose, pose_similarity
except Exception:
    RETRIEVAL_AVAILABLE = False


# ====================== utils: image I/O & drawing ======================
def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def pil_from_bgr(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def bgr_from_pil(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def fit_into_box(img_bgr: np.ndarray, box_wh: Tuple[int, int], pad_color=(245, 245, 245)) -> np.ndarray:
    """Letterbox into (w,h) while keeping aspect; pads with pad_color."""
    w, h = box_wh
    ih, iw = img_bgr.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), pad_color, dtype=np.uint8)
    x1 = (w - nw) // 2
    y1 = (h - nh) // 2
    canvas[y1:y1 + nh, x1:x1 + nw] = resized
    return canvas

def draw_text(pil_img: Image.Image, text: str, xy=(10, 10), fill=(20, 20, 20), size=22):
    """Robust text drawing with fallback font."""
    draw = ImageDraw.Draw(pil_img)
    try:
        # Windows users usually have Arial; replace with your font if needed (e.g., "simhei.ttf" for CJK).
        font = ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=fill, font=font)


# ====================== data structures ======================
@dataclass
class MatchItem:
    path: str
    clip_score: float
    pose_score: Optional[float]
    fused_score: float


# ====================== montage builder ======================
def make_montage(
    input_path: str,
    matches: List[MatchItem],
    out_path: str,
    *,
    canvas_w: int = 1800,
    header_h: int = 72,
    left_w: int = 520,
    cell_w: int = 420,
    cell_h: int = 320,
    pad: int = 16,
    cols: int = 2,
    bg=(255, 255, 255)
) -> str:
    """
    Layout:
    ┌──────────── canvas_w ────────────┐
    │  Header (title + input path)     │  header_h
    ├──── left_w ────┬──── right area ─┤
    │  Input image   │  grid of matches│
    │                │  (cols columns) │
    └────────────────┴─────────────────┘
    """
    rows = max(1, int(math.ceil(len(matches) / max(1, cols))))
    right_w = canvas_w - left_w - pad * 3
    grid_w = right_w
    grid_h = rows * cell_h + (rows + 1) * pad

    left_h = grid_h
    canvas_h = header_h + pad + max(left_h, grid_h) + pad

    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)

    # Header
    pil_canvas = pil_from_bgr(canvas)
    title = "Embodied Aesthetic Retrieval — Montage"
    info = f"Input: {os.path.basename(input_path)} | Top-{len(matches)}"
    draw_text(pil_canvas, title, (pad, 12), size=28)
    draw_text(pil_canvas, info, (pad, 42), size=20)
    canvas = bgr_from_pil(pil_canvas)

    # Left: input image
    try:
        inp_bgr = read_bgr(input_path)
    except Exception:
        inp_bgr = cv2.cvtColor(np.array(Image.open(input_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    inp_fit = fit_into_box(inp_bgr, (left_w, left_h))
    canvas[header_h + pad:header_h + pad + left_h, pad:pad + left_w] = inp_fit

    # Label "Input"
    pil_canvas = pil_from_bgr(canvas)
    draw_text(pil_canvas, "Input", (pad + 10, header_h + pad + 10), size=20, fill=(70, 70, 70))
    canvas = bgr_from_pil(pil_canvas)

    # Right: matches grid
    x0 = pad * 2 + left_w
    y0 = header_h + pad

    for idx, m in enumerate(matches):
        r = idx // cols
        c = idx % cols
        gx = x0 + c * (cell_w + pad)
        gy = y0 + r * (cell_h + pad)

        try:
            mg = read_bgr(m.path)
        except Exception:
            mg = cv2.cvtColor(np.array(Image.open(m.path).convert("RGB")), cv2.COLOR_RGB2BGR)

        # main image area (reserve 46px footer bar)
        mg_fit = fit_into_box(mg, (cell_w, cell_h - 46))
        canvas[gy:gy + cell_h - 46, gx:gx + cell_w] = mg_fit

        # footer bar with scores
        bar = np.full((46, cell_w, 3), (250, 250, 250), dtype=np.uint8)
        pil_bar = pil_from_bgr(bar)
        fname = os.path.basename(m.path)
        clip_s = f"CLIP={m.clip_score:.3f}"
        pose_s = f" | Pose={m.pose_score:.3f}" if m.pose_score is not None else ""
        fused_s = f" | Fused={m.fused_score:.3f}"
        draw_text(pil_bar, f"{fname} | {clip_s}{pose_s}{fused_s}", (8, 10), size=18, fill=(40, 40, 40))
        bar = bgr_from_pil(pil_bar)
        canvas[gy + cell_h - 46:gy + cell_h, gx:gx + cell_w] = bar

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)
    return out_path


# ====================== retrieval mode ======================
def iter_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                yield os.path.join(root, f)

def run_retrieval_and_export(folder: str, index_dir: str, outdir: str, topk: int, alpha: float,
                             cols: int, canvas_w: int, cell_w: int, cell_h: int):
    if not RETRIEVAL_AVAILABLE:
        raise RuntimeError("Retrieval modules not available. "
                           "Run from project root so 'processors' can be imported.")

    from processors.retrieval import load_clip, embed_image, CorpusIndex, fuse_scores
    from processors.pose import extract_pose, pose_similarity

    model, preprocess, device = load_clip()
    idx = CorpusIndex(index_dir)

    for ipath in tqdm(list(iter_images(folder)), desc="Processing"):
        try:
            img = Image.open(ipath).convert("RGB")
        except Exception as e:
            print("skip (bad image):", ipath, e)
            continue

        qfeat = embed_image(model, preprocess, device, img)
        qpose = extract_pose(img)
        qpose_kps = qpose.keypoints if qpose.ok else None

        sims, ids = idx.search(qfeat, topk=topk * 3)
        items: List[MatchItem] = []
        for s, i in zip(sims, ids):
            path = idx.paths[i]
            ps = None
            if qpose_kps is not None:
                cpose = idx.poses[i]
                if cpose is not None:
                    ps = pose_similarity(qpose_kps, cpose)
            fused = fuse_scores(float(s), ps, alpha)
            items.append(MatchItem(path=path, clip_score=float(s), pose_score=ps, fused_score=fused))

        items.sort(key=lambda x: x.fused_score, reverse=True)
        items = items[:topk]

        outname = os.path.splitext(os.path.basename(ipath))[0] + f"_top{topk}.jpg"
        outpath = os.path.join(outdir, outname)
        make_montage(ipath, items, outpath, cols=cols, canvas_w=canvas_w, cell_w=cell_w, cell_h=cell_h)


# ====================== CSV mode ======================
def load_csv(csv_path: str) -> Dict[str, List[MatchItem]]:
    """
    CSV columns: input_image,rank,match_path,clip_score,pose_score,fused_score
    Returns: mapping input_image -> [MatchItem... ranked]
    """
    mapping: Dict[str, List[MatchItem]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inp = row["input_image"]
            clip_s = float(row["clip_score"]) if row["clip_score"] else 0.0
            pose_s = float(row["pose_score"]) if row["pose_score"] else None
            fused_s = float(row["fused_score"]) if row["fused_score"] else clip_s
            item = MatchItem(path=row["match_path"], clip_score=clip_s, pose_score=pose_s, fused_score=fused_s)
            mapping.setdefault(inp, []).append(item)
    # ensure ranked order
    for k in mapping:
        mapping[k].sort(key=lambda x: x.fused_score, reverse=True)
    return mapping

def export_from_csv(csv_path: str, outdir: str, topk: int, cols: int, canvas_w: int, cell_w: int, cell_h: int):
    mapping = load_csv(csv_path)
    for ipath, items in tqdm(mapping.items(), desc="Rendering"):
        chosen = items[:topk]
        outname = os.path.splitext(os.path.basename(ipath))[0] + f"_top{topk}.jpg"
        outpath = os.path.join(outdir, outname)
        make_montage(ipath, chosen, outpath, cols=cols, canvas_w=canvas_w, cell_w=cell_w, cell_h=cell_h)


# ====================== main ======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", help="Input image folder (run retrieval and export)")
    ap.add_argument("--csv", help="Use existing CSV (batch_results.csv) to export montage only")
    ap.add_argument("--index", default="data/indices/paintings", help="Index directory (for --folder mode)")
    ap.add_argument("--outdir", default="montages", help="Output folder for montages")
    ap.add_argument("--topk", type=int, default=5, help="Top-K per montage")
    ap.add_argument("--alpha", type=float, default=0.8, help="Fusion weight (clip vs pose) for --folder mode")

    # Layout controls (optional)
    ap.add_argument("--cols", type=int, default=2, help="Columns for match grid")
    ap.add_argument("--canvas_w", type=int, default=1800, help="Canvas width")
    ap.add_argument("--cell_w", type=int, default=420, help="Match cell width")
    ap.add_argument("--cell_h", type=int, default=320, help="Match cell height")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.csv:
        export_from_csv(args.csv, args.outdir, args.topk, args.cols, args.canvas_w, args.cell_w, args.cell_h)
    elif args.folder:
        run_retrieval_and_export(args.folder, args.index, args.outdir, args.topk, args.alpha,
                                 args.cols, args.canvas_w, args.cell_w, args.cell_h)
    else:
        print("Please provide --folder or --csv")
        return

    print(f"[done] Montages saved to: {args.outdir}")

if __name__ == "__main__":
    main()
