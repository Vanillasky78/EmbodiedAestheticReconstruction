# processors/matching.py
import cv2, numpy as np
from PIL import Image
from typing import Tuple
from sklearn.cluster import KMeans
from skimage.color import deltaE_ciede2000

# ---------- composition ----------
def thirds_heatmap(w:int, h:int) -> np.ndarray:
    cx = np.array([w/3, 2*w/3]); cy = np.array([h/3, 2*h/3])
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.zeros((h,w), np.float32)
    for x0 in cx:
        for y0 in cy:
            heat += np.exp(-(((xx-x0)**2+(yy-y0)**2)/(2*(0.12*min(w,h))**2)))
    heat /= heat.max()+1e-8
    return heat

def subject_box(pil: Image.Image) -> Tuple[int,int,int,int]:
    bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, m = sal.computeSaliency(bgr)
    m = (m*255).astype(np.uint8)
    _, th = cv2.threshold(m,0,255,cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0,0,bgr.shape[1]-1,bgr.shape[0]-1
    x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return x,y,x+w,y+h

def composition_score(pil_q: Image.Image, pil_ref: Image.Image)->float:
    def score_one(pil):
        w,h = pil.size
        heat = thirds_heatmap(w,h)
        x1,y1,x2,y2 = subject_box(pil)
        mask = np.zeros((h,w), np.uint8)
        mask[y1:y2, x1:x2] = 1
        return float((heat*mask).sum() / (mask.sum()+1e-8))
    s1, s2 = score_one(pil_q), score_one(pil_ref)
    return float(np.exp(-4.0*abs(s1-s2)))

# ---------- palette ----------
def palette_score(pil_q: Image.Image, pil_ref: Image.Image, k:int=5)->float:
    def centers(pil):
        arr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2LAB)
        flat = arr.reshape(-1,3).astype(np.float32)
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(flat)
        return km.cluster_centers_
    c1, c2 = centers(pil_q), centers(pil_ref)
    de = deltaE_ciede2000(c1[:min(len(c1),len(c2))][None,...], c2[:min(len(c1),len(c2))][None,...])[0]
    return float(np.exp(-0.06*float(np.mean(de))))

# ---------- limb angle histogram ----------
def limb_hist_score(kps_a: np.ndarray, kps_b: np.ndarray)->float:
    pairs = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]
    def angles(kps):
        ang=[]
        for i,j in pairs:
            if i<len(kps) and j<len(kps):
                v = kps[j]-kps[i]
                if np.linalg.norm(v)>1e-6:
                    ang.append(np.arctan2(v[1], v[0]))
        if not ang: return None
        hist,_ = np.histogram(ang, bins=18, range=(-np.pi,np.pi), density=True)
        return hist
    ha, hb = angles(kps_a), angles(kps_b)
    if ha is None or hb is None: return 0.0
    return float(np.minimum(ha,hb).sum())
