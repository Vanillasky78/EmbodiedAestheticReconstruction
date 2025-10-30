# processors/camera.py
# -------------------------------------------------------------
# Auto-detect camera (0/1/2/3), pick best resolution, and
# capture a frame after N seconds of stillness.
#
# Public API:
#   find_working_camera(indices=(0,1,2,3))
#   capture_locked_frame(out_path: str, still_secs=3.0, ...)
#
# Notes (macOS permissions):
#   If no camera opens, go to:
#   System Settings → Privacy & Security → Camera
#   and allow Terminal / VS Code / Safari (if used).
# -------------------------------------------------------------

from __future__ import annotations
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


# ---------- Low-level helpers ----------

def _try_open(index: int, warmup_frames: int = 10) -> Optional[cv2.VideoCapture]:
    """Try to open a camera device; return cv2.VideoCapture or None."""
    cap = cv2.VideoCapture(index)
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None

    # quick warmup to stabilize exposure/white balance
    for _ in range(warmup_frames):
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return None
    return cap


def _set_best_resolution(cap: cv2.VideoCapture, candidates: Iterable[Tuple[int, int]] = ((1920,1080),(1280,720),(960,540),(640,480))) -> Tuple[int,int]:
    """Try to set the best (highest) working resolution; return (w, h) actually set."""
    chosen = (0, 0)
    for w, h in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
        rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if rw == w and rh == h:
            chosen = (rw, rh)
            break
        # keep last successful (may be 0,0 initially)
        if rw*rh > chosen[0]*chosen[1]:
            chosen = (rw, rh)
    return chosen


# ---------- Public: camera discovery ----------

def find_working_camera(indices: Iterable[int] = (0, 1, 2, 3)) -> Tuple[int, cv2.VideoCapture]:
    """
    Try a list of indices and return (index, cap) for the first working device.
    Raises RuntimeError if none opens.
    """
    last_err = None
    for idx in indices:
        cap = _try_open(idx)
        if cap is not None:
            _set_best_resolution(cap)
            return idx, cap
        last_err = idx

    raise RuntimeError(
        "Could not open camera on indices {}.\n"
        "• On macOS, check System Settings → Privacy & Security → Camera and allow Terminal/VS Code/Safari.\n"
        "• If you use an external USB camera, plug it in and retry.\n"
        "• You may also try a different index manually (e.g., 1).".format(list(indices))
    )


# ---------- Public: stillness-based capture ----------

def capture_locked_frame(
    out_path: str = "data/interim/locked_frame.jpg",
    still_secs: float = 3.0,
    sample_hz: float = 10.0,
    motion_thresh: float = 2.0,   # smaller = more sensitive to motion
    blur_ksize: int = 5,
    indices_to_try: Iterable[int] = (0, 1, 2, 3),
    show_debug_window: bool = False,
) -> str:
    """
    Open a camera (auto-detect), wait until scene is still for `still_secs`,
    save a frame to `out_path`, and return the path.

    Heuristic:
      - Compute per-frame grayscale absdiff.
      - Use mean absolute difference as motion score.
      - When score stays below `motion_thresh` for `still_secs`, capture.

    Parameters
    ----------
    out_path : str
        Output JPEG path to save.
    still_secs : float
        Required continuous stillness duration.
    sample_hz : float
        Processing frequency (read every 1/sample_hz seconds).
    motion_thresh : float
        Motion score threshold (0~255). Typical 1~4 works well.
    blur_ksize : int
        Gaussian blur kernel size to suppress noise before diff.
    indices_to_try : Iterable[int]
        Camera indices to try in order (default 0,1,2,3).
    show_debug_window : bool
        If True, pops an OpenCV window with live motion score (for local debug).

    Returns
    -------
    str
        Saved image path.

    Raises
    ------
    RuntimeError
        If camera cannot be opened.
    """
    # ensure output dir
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cam_idx, cap = find_working_camera(indices_to_try)

    # optional: display which camera we picked
    print(f"[camera] Using device index: {cam_idx}  @ {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    prev_gray = None
    still_start: Optional[float] = None
    period = 1.0 / max(1e-6, sample_hz)

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Failed to read from camera.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if blur_ksize > 1:
                gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

            motion_score = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_score = float(np.mean(diff))
            prev_gray = gray

            # check stillness timeline
            if motion_score <= motion_thresh:
                still_start = still_start or time.time()
            else:
                still_start = None

            if show_debug_window:
                vis = frame.copy()
                cv2.putText(vis, f"motion={motion_score:.2f}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if still_start:
                    elapsed = time.time() - still_start
                    cv2.putText(vis, f"still {elapsed:.1f}/{still_secs:.1f}s", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
                cv2.imshow("EAR Camera (debug)", vis)
                # Allow ESC to abort
                if (cv2.waitKey(1) & 0xFF) == 27:
                    raise KeyboardInterrupt

            if still_start and (time.time() - still_start) >= still_secs:
                # capture!
                ok = cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not ok:
                    raise RuntimeError(f"Failed to write image: {out_path}")
                return out_path

            # throttle loop
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    finally:
        try:
            cap.release()
        except Exception:
            pass
        if show_debug_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
