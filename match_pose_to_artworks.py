# match_pose_to_artworks.py
# ------------------------------------------------------------
# End-to-end helper to:
# 1) Capture a still frame from camera (3s stillness).
# 2) Search the curated portrait index (FAISS) using the same
#    multi-cue matcher as the Streamlit app (CLIP + Pose + Color).
# 3) Return the top-K results and useful paths for UI rendering.
#
# Usage (CLI test):
#   python match_pose_to_artworks.py --topk 6 --no-public-domain
#
# Usage (from app.py):
#   from match_pose_to_artworks import run_full_capture_and_match
#   result = run_full_capture_and_match(topk=12, require_public_domain=True)
#
# Output (dict):
# {
#   "locked_frame_path": "data/interim/locked_frame.jpg",
#   "results": [
#       {"rid": "...", "score": 0.812, "meta": {...}, "image_path": "..."},
#       ...
#   ],
#   "device": "mps/cuda/cpu",
#   "index_info": {"n_items": int}
# }

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

# ML + ANN
import torch
import open_clip
import faiss

# local utils
from processors.camera import capture_locked_frame
from processors.retrieval import Matcher

# --------------------------
# Paths (keep consistent with app.py / indexing)
# --------------------------
INDEX_DIR  = Path("indexing")
INDEX_PATH = INDEX_DIR / "faiss.index"
IDS_PATH   = INDEX_DIR / "ids.npy"
META_PATH  = INDEX_DIR / "meta.json"

INTERIM_DIR = Path("data") / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
LOCKED_FRAME = INTERIM_DIR / "locked_frame.jpg"


# --------------------------
# Lazy-load CLIP + FAISS
# (与 app.py 同步，避免重复代码差异)
# --------------------------
def _load_matcher() -> Tuple[Matcher, str, Dict[str, int]]:
    """Load CLIP model + FAISS index + metadata, return a ready Matcher and device."""
    device = "cuda" if torch.cuda.is_available() else (
             "mps"  if torch.backends.mps.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval().to(device)

    # 必要文件检查
    missing = [p for p in (INDEX_PATH, IDS_PATH, META_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Index not found. Please build it first:\n"
            "  python indexing/build_index.py \\\n"
            "    --dataset_jsonl data/interim/portrait_art_dataset.jsonl \\\n"
            "    --images_dir    data/images\n"
            f"Missing: {', '.join(str(p) for p in missing)}"
        )

    # 载入 FAISS + 元数据
    index = faiss.read_index(str(INDEX_PATH))
    ids   = np.load(str(IDS_PATH), allow_pickle=True)
    meta  = json.loads(META_PATH.read_text(encoding="utf-8"))

    if len(ids) != len(meta):
        # 不致命，给个提醒
        print(f"[warn] Index/meta length mismatch: ids={len(ids)} meta={len(meta)}")

    matcher = Matcher(index, ids, meta, model, preprocess, device=device)
    info = {"n_items": int(index.ntotal)}
    return matcher, device, info


# --------------------------
# Public API
# --------------------------
def run_full_capture_and_match(
    topk: int = 12,
    require_public_domain: bool = True,
    w_clip: float = 0.60,
    w_pose: float = 0.30,
    w_color: float = 0.10,
    k_search: int = 200,
    save_locked_frame_to: Path | str = LOCKED_FRAME,
) -> Dict[str, Any]:
    """
    抓拍 → 检索 的一站式流程。

    Parameters
    ----------
    topk : int
        返回的前 K 个结果（UI 展示数量）
    require_public_domain : bool
        是否只返回 Public Domain / CC0
    w_clip, w_pose, w_color : float
        三种相似度的权重（与 app.py 的侧边栏一致）
    k_search : int
        FAISS 初筛取多少个候选后再综合重排
    save_locked_frame_to : Path | str
        抓取的静止帧保存路径

    Returns
    -------
    dict:
        {
          "locked_frame_path": <str>,
          "results": [{"rid":..., "score":..., "meta":{...}, "image_path":...}, ...],
          "device": <"cuda"/"mps"/"cpu">,
          "index_info": {"n_items": int}
        }
    """
    save_locked_frame_to = Path(save_locked_frame_to)
    save_locked_frame_to.parent.mkdir(parents=True, exist_ok=True)

    # 1) 摄像头抓帧（已内置：自动轮询 0/1/2/3、3s 静止检测、失败会抛异常）
    locked_frame_path = capture_locked_frame(out_path=str(save_locked_frame_to))

    # 2) 载入检索器（CLIP + FAISS + metadata）
    matcher, device, index_info = _load_matcher()

    # 3) 读取图片 → 做多模态检索
    q_img = Image.open(locked_frame_path).convert("RGB")
    weights = dict(w_clip=w_clip, w_pose=w_pose, w_color=w_color)
    results = matcher.search(
        q_img,
        k=k_search,
        weights=weights,
        filters={"require_public_domain": require_public_domain},
        topn=topk,
    )

    # 4) 组装更易用的结构（供 UI 渲染）
    packed: List[Dict[str, Any]] = []
    for rid, meta, score in results:
        # meta 中常见键：artwork_title_en/artist_name_en/year/museum/license/image_path ...
        packed.append(
            {
                "rid": str(rid),
                "score": float(score),
                "meta": meta,
                "image_path": meta.get("image_path"),
            }
        )

    return {
        "locked_frame_path": str(locked_frame_path),
        "results": packed,
        "device": device,
        "index_info": index_info,
    }


# --------------------------
# CLI for quick manual test
# --------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Capture from camera and match against portrait index."
    )
    p.add_argument("--topk", type=int, default=6, help="How many results to return.")
    p.add_argument(
        "--no-public-domain",
        action="store_true",
        help="If set, do NOT require Public Domain/CC0.",
    )
    p.add_argument("--w-clip", type=float, default=0.60)
    p.add_argument("--w-pose", type=float, default=0.30)
    p.add_argument("--w-color", type=float, default=0.10)
    p.add_argument("--k-search", type=int, default=200)
    p.add_argument(
        "--save-to", type=str, default=str(LOCKED_FRAME),
        help="Locked frame output path."
    )
    return p


def main():
    args = _build_argparser().parse_args()
    out = run_full_capture_and_match(
        topk=args.topk,
        require_public_domain=(not args.no_public_domain),
        w_clip=args.w_clip,
        w_pose=args.w_pose,
        w_color=args.w_color,
        k_search=args.k_search,
        save_locked_frame_to=args.save_to,
    )

    print(f"\n[ok] Locked frame saved to: {out['locked_frame_path']}")
    print(f"[ok] Device: {out['device']} | Index items: {out['index_info']['n_items']}")
    print(f"[ok] Top-{len(out['results'])} results:")
    for i, r in enumerate(out["results"], 1):
        meta = r["meta"]
        title  = meta.get("artwork_title_en") or meta.get("title") or "Untitled"
        artist = meta.get("artist_name_en") or meta.get("artistDisplayName") or "Unknown"
        year   = meta.get("year") or meta.get("objectDate") or "?"
        museum = meta.get("museum") or meta.get("department") or ""
        print(f"  {i:>2}. {title} — {artist} • {year} • {museum} | score={r['score']:.3f}")


if __name__ == "__main__":
    main()
