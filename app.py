# app.py
import os, cv2, numpy as np, tempfile
from PIL import Image
import streamlit as st

from processors.pose import extract_pose
from processors.retrieval import load_clip, embed_image, CorpusIndex
from processors.visual import procrustes_pose_score, face_angle_score, silhouette_ssim, color_corr
from processors.fusion import fuse_artistic

# ---- Load indices ----
@st.cache_resource
def load_indices():
    ROOT=os.path.dirname(os.path.abspath(__file__))
    inds={}
    p_museum=os.path.join(ROOT,"data","indices","paintings")
    if os.path.exists(p_museum): inds[("Museum","All")]=CorpusIndex(p_museum)
    fashion_root=os.path.join(ROOT,"data","indices_fashion")
    if os.path.isdir(fashion_root):
        for brand in os.listdir(fashion_root):
            p=os.path.join(fashion_root,brand)
            if os.path.isdir(p): inds[("Fashion",brand)]=CorpusIndex(p)
    return inds

model, preprocess, device = load_clip()
INDICES=load_indices()

st.set_page_config(page_title="Embodied Aesthetic", layout="wide")
st.title("Embodied Aesthetic — Multi-signal Matching & Fusion")

colL,colM,colR=st.columns([1.15,1.45,1.4])

# ---- Left controls ----
with colL:
    st.subheader("1) Input & Controls")
    uploaded=st.file_uploader("Upload an image",type=["png","jpg","jpeg","webp"])
    domain=st.selectbox("Domain",["Museum","Fashion"])
    brands=["All"] if domain=="Museum" else sorted([b for (d,b) in INDICES.keys() if d=="Fashion"])
    brand=st.selectbox("Brand",brands)
    topk=st.slider("Top-K",1,12,6)
    st.markdown("**Re-ranking Weights**")
    w_clip=st.slider("w_clip (semantic)",0.0,1.0,0.45,0.05)
    w_pose=st.slider("w_pose (pose)",0.0,1.0,0.20,0.05)
    w_face=st.slider("w_face (orientation)",0.0,1.0,0.15,0.05)
    w_sil=st.slider("w_sil (silhouette)",0.0,1.0,0.10,0.05)
    w_color=st.slider("w_color (color)",0.0,1.0,0.10,0.05)
    find_btn=st.button("Find Matches",type="primary")

# ---- Middle: results ----
results=[]
qimg=None
with colM:
    st.subheader("2) Matches (Re-ranked)")
    key=(domain,brand if domain=="Fashion" else "All")
    if key not in INDICES:
        st.warning("Index not available.")
    elif uploaded and find_btn:
        qimg=Image.open(uploaded).convert("RGB")
        st.image(qimg,caption="Query",use_container_width=True)
        idx=INDICES[key]
        qfeat=embed_image(model,preprocess,device,qimg)
        sims,ids=idx.search(qfeat,topk=max(topk*5,topk))
        qpose=extract_pose(qimg)
        re_rank=[]
        for s,i in zip(sims,ids):
            path=idx.paths[i]
            clip_s=float(s)
            pose_s=None
            if qpose.ok and idx.poses[i] is not None:
                pose_s=procrustes_pose_score(qpose.keypoints,idx.poses[i])
            ref_pil=Image.open(path).convert("RGB")
            face_s=face_angle_score(qimg,ref_pil)
            sil_s=silhouette_ssim(qimg,ref_pil)
            color_s=color_corr(qimg,ref_pil)
            final=(w_clip*clip_s
                   +w_pose*(pose_s or 0)
                   +w_face*face_s
                   +w_sil*sil_s
                   +w_color*color_s)
            re_rank.append((path,clip_s,pose_s,face_s,sil_s,color_s,final))
        re_rank.sort(key=lambda x:x[-1],reverse=True)
        results=re_rank[:topk]
        for (path,cs,ps,fs,ss,cc,fin) in results:
            caption=f"Fused={fin:.3f} | CLIP={cs:.3f}"
            if ps is not None: caption+=f" | Pose={ps:.3f}"
            caption+=f" | Face={fs:.3f} | Sil={ss:.3f} | Color={cc:.3f}"
            st.image(path,caption=caption,use_container_width=True)

# ---- Right: fusion ----
with colR:
    st.subheader("3) Fusion Result")
    if qimg and results:
        ref_path=results[0][0]
        tmp=tmpfile=tempfile.mktemp(suffix=".jpg")
        fused=fuse_artistic(uploaded.name,ref_path,tmp)
        st.image(fused,caption="Artistic Fusion",use_container_width=True)
