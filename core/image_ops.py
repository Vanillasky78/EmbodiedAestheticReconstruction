import numpy as np, cv2
from PIL import Image

def _pil_to_cv(img: Image.Image):
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def _cv_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def stylise_image(img: Image.Image, style_name: str, strength: float) -> Image.Image:
    bgr = _pil_to_cv(img)
    h, w = bgr.shape[:2]

    if style_name == "Organic Flux":
        smooth = cv2.edgePreservingFilter(bgr, flags=1, sigma_s=30, sigma_r=0.35)
        bilat = cv2.bilateralFilter(smooth, 9, 75, 75)
        out = cv2.addWeighted(bilat, 0.8, smooth, 0.2, 0)
    elif style_name == "Geo Deconstruct":
        edges, _ = cv2.pencilSketch(bgr, sigma_s=60, sigma_r=0.07, shade_factor=0.04)
        color = cv2.edgePreservingFilter(bgr, flags=2, sigma_s=60, sigma_r=0.2)
        quant = (color // 32) * 32
        out = cv2.addWeighted(quant, 0.85, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.15, 0)
    elif style_name == "Noisy Bloom":
        blur = cv2.GaussianBlur(bgr, (0, 0), 8)
        bloom = cv2.addWeighted(bgr, 0.6, blur, 0.4, 0)
        noise = np.random.normal(0, 8, size=bloom.shape).astype(np.float32)
        out = np.clip(bloom.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif style_name == "Monochrome Ritual":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mono = cv2.equalizeHist(gray)
        mono = cv2.normalize(mono, None, 0, 255, cv2.NORM_MINMAX)
        mono_bgr = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        Y, X = np.ogrid[:h, :w]; cx, cy = w/2, h/2; rx, ry = w*0.6, h*0.6
        mask = ((X-cx)**2)/(rx**2) + ((Y-cy)**2)/(ry**2)
        mask = np.clip(mask, 0, 1)
        out = (mono_bgr * (1 - 0.35 * mask[..., None])).astype(np.uint8)
    else:  # "Chromatic Drift"
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + 15) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.1, 0, 255)
        drift = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        grad = np.linspace(0.85, 1.0, w, dtype=np.float32); grad = np.tile(grad, (h, 1))
        out = (drift.astype(np.float32) * grad[..., None]).clip(0, 255).astype(np.uint8)

    a = float(np.clip(strength, 0.0, 1.0))
    blended = cv2.addWeighted(out, a, bgr, 1 - a, 0)
    return _cv_to_pil(blended)
