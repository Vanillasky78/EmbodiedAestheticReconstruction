# processors/fusion_advanced.py
import cv2, numpy as np
from PIL import Image
from .fusion import read_rgb, save_rgb, color_transfer

def person_matte(pil_img: Image.Image) -> np.ndarray:
    import mediapipe as mp
    im = np.array(pil_img.convert("RGB"))
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        m = seg.process(cv2.cvtColor(im, cv2.COLOR_RGB2BGR)).segmentation_mask
    return np.clip((m-0.35)/0.4, 0, 1).astype(np.float32)

def skin_mask(rgb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[:,:,1], ycrcb[:,:,2]
    return ((cr>135)&(cr<180)&(cb>85)&(cb<135)).astype(np.uint8)

def pyramid_blend(fg: np.ndarray, bg: np.ndarray, mask: np.ndarray, levels:int=4) -> np.ndarray:
    mask3 = np.dstack([mask]*3).astype(np.float32)
    gp_fg, gp_bg, gp_m = [fg.astype(np.float32)], [bg.astype(np.float32)], [mask3]
    for _ in range(levels):
        gp_fg.append(cv2.pyrDown(gp_fg[-1])); gp_bg.append(cv2.pyrDown(gp_bg[-1])); gp_m.append(cv2.pyrDown(gp_m[-1]))
    lp_fg, lp_bg = [gp_fg[-1]], [gp_bg[-1]]
    for i in range(levels,0,-1):
        size = (gp_fg[i-1].shape[1], gp_fg[i-1].shape[0])
        Lf = gp_fg[i-1] - cv2.pyrUp(gp_fg[i], dstsize=size)
        Lb = gp_bg[i-1] - cv2.pyrUp(gp_bg[i], dstsize=size)
        lp_fg.append(Lf); lp_bg.append(Lb)
    out = lp_fg[0]*gp_m[-1] + lp_bg[0]*(1-gp_m[-1])
    for i in range(1,levels+1):
        out = cv2.pyrUp(out, dstsize=(lp_fg[i].shape[1], lp_fg[i].shape[0]))
        m = gp_m[levels-i]
        out = out*m + (lp_bg[i]*(1-m) + lp_fg[i]*m)
    return np.clip(out,0,255).astype(np.uint8)

def soft_shadow(mask: np.ndarray, shift=(12,12), blur=25, strength=0.35) -> np.ndarray:
    m = (mask*255).astype(np.uint8)
    M = cv2.copyMakeBorder(cv2.GaussianBlur(m,(blur|1, blur|1), blur/3),
                           shift[1],0,shift[0],0, cv2.BORDER_CONSTANT, value=0)
    M = M[:mask.shape[0], :mask.shape[1]]
    return (M.astype(np.float32)/255.0)*strength

def integrate_into_artwork(participant_path: str, artwork_path: str, out_path: str=None) -> np.ndarray:
    q = Image.open(participant_path).convert("RGB")
    a = Image.open(artwork_path).convert("RGB")
    fg = np.array(q); bg = np.array(a.resize((fg.shape[1], fg.shape[0])))
    alpha = person_matte(q)
    skin = skin_mask(fg)
    lab_fg = cv2.cvtColor(fg, cv2.COLOR_RGB2LAB)
    lab_bg = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB)
    Lf2 = (lab_fg[:,:,0].astype(np.float32)-lab_fg[:,:,0].mean())/(lab_fg[:,:,0].std()+1e-6)*(lab_bg[:,:,0].std()+1e-6)+lab_bg[:,:,0].mean()
    lab_fg[:,:,0] = np.clip(Lf2,0,255).astype(np.uint8)
    fg_harmon = cv2.cvtColor(lab_fg, cv2.COLOR_LAB2RGB)
    strong = color_transfer(fg,bg)
    mask_nonskin = (1-skin).astype(np.float32)[...,None]
    fg_region = (fg_harmon*(1-mask_nonskin) + strong*mask_nonskin).astype(np.uint8)
    blended = pyramid_blend(fg_region,bg,alpha,levels=4)
    sh = soft_shadow(alpha, shift=(int(0.01*fg.shape[1]), int(0.015*fg.shape[0])), blur=31, strength=0.25)
    blended = (blended.astype(np.float32)*(1-sh[...,None])).astype(np.uint8)
    if out_path: save_rgb(blended,out_path)
    return blended
