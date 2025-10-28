"""
match_pose_to_artworks.py
----------------------------------------
Full pipeline for pose-to-portrait retrieval.

Flow:
1. Live camera feed watches the participant.
2. When the participant holds a pose (low movement) for ~3 seconds,
   the system "locks" that frame.
3. We extract body keypoints using YOLOv8-Pose and render a clean
   skeleton image (black background, white joints/limbs).
4. We encode that skeleton image using OpenCLIP to get a pose embedding.
5. We compare that embedding to the reference portrait dataset
   (20 curated artworks) and return the Top-3 closest matches.

This script is intended to be called in installation mode
or imported by a Streamlit app for display.
"""

import os
import numpy as np
import torch
from PIL import Image

from processors.camera import capture_locked_frame
from processors.pose import PoseEstimator, make_pose_detector
from processors.retrieval import ArtworkDatabase
from open_clip import create_model_and_transforms


# ---------------------------
# 1. Configuration
# ---------------------------

CSV_PATH = "data/portrait_works.csv"
RAW_SAVE_DIR = "data/interim/raw_frames"
POSE_SAVE_DIR = "data/interim/pose_frames"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def encode_image_with_clip(img_path, clip_model, preprocess, device):
    """
    Load an image, preprocess it for OpenCLIP, and return
    a normalized embedding as a NumPy vector.
    """
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = clip_model.encode_image(img_tensor)
        emb /= emb.norm(dim=-1, keepdim=True)

    return emb.cpu().numpy().reshape(-1)  # (D,)


def run_full_capture_and_match():
    """
    Run the full pipeline once:
    - wait for subject to hold still
    - capture frame
    - generate skeleton
    - encode pose
    - retrieve Top-3 artworks

    Returns:
        {
            "locked_frame_path": str,
            "skeleton_path": str,
            "results": [ {rank, artist, title, year, score, notes_pose, file_name}, ... ]
        }
    """

    print(f"🔥 Using device: {DEVICE}")

    # ---------------------------
    # 2. Init models
    # ---------------------------
    print("Initializing pose estimator (YOLOv8-Pose)...")
    pose_estimator = PoseEstimator(device=DEVICE)
    pose_detector = make_pose_detector(pose_estimator)

    print("Initializing OpenCLIP model...")
    clip_model, preprocess, _ = create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()

    print("Loading portrait artwork database...")
    db = ArtworkDatabase(csv_path=CSV_PATH)

    # ---------------------------
    # 3. Camera: wait for stillness
    # ---------------------------
    print("\n📷 Camera is live.")
    print("Please hold your pose (very still) for about 3 seconds.\n")

    locked_frame_path = capture_locked_frame(
        pose_detector=pose_detector,
        save_dir=RAW_SAVE_DIR,
        stable_seconds=3.0,
        movement_threshold=5.0,
        camera_index=0,
        show_debug_window=False  # set True while developing
    )

    print(f"✅ Captured locked frame: {locked_frame_path}")

    # ---------------------------
    # 4. Create skeleton pose image from locked frame
    # ---------------------------
    # derive a timestamp-like suffix for naming
    base_name = os.path.splitext(os.path.basename(locked_frame_path))[0]
    skeleton_path = os.path.join(
        POSE_SAVE_DIR,
        f"{base_name}_skeleton.png"
    )

    # load frame as RGB array for pose processing
    frame_rgb = np.array(Image.open(locked_frame_path).convert("RGB"))

    keypoints = pose_estimator.extract_keypoints(frame_rgb)
    if keypoints is None:
        raise RuntimeError("No person detected in the locked frame. Cannot continue.")

    pose_estimator.draw_skeleton_image(
        frame_bgr=frame_rgb[:, :, ::-1],  # convert RGB to BGR for drawing consistency
        keypoints=keypoints,
        save_path=skeleton_path
    )
    print(f"✅ Saved skeleton pose image: {skeleton_path}")

    # ---------------------------
    # 5. Encode participant pose as embedding
    # ---------------------------
    print("Encoding participant pose with OpenCLIP...")
    user_embedding = encode_image_with_clip(
        img_path=skeleton_path,
        clip_model=clip_model,
        preprocess=preprocess,
        device=DEVICE
    )

    # ---------------------------
    # 6. Retrieve Top-3 closest artworks
    # ---------------------------
    print("Retrieving Top-3 portrait matches...")
    results = db.retrieve_top_k(user_embedding, k=3)

    print("\n🎯 Top matches:\n")
    for item in results:
        print(f"{item['rank']}. {item['artist']} — {item['title']} ({item['year']})")
        print(f"   Similarity score: {item['score']:.4f}")
        print(f"   Pose reading: {item['notes_pose']}")
        print()

    # ---------------------------
    # 7. Return structured result (for Streamlit / UI layer)
    # ---------------------------
    return {
        "locked_frame_path": locked_frame_path,
        "skeleton_path": skeleton_path,
        "results": results
    }


if __name__ == "__main__":
    output = run_full_capture_and_match()
    print("Done.")
