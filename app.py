import streamlit as st
from PIL import Image
from processors.pose import extract_pose, pose_similarity
from processors.retrieval import load_clip, embed_image, CorpusIndex, fuse_scores

@st.cache_resource
def load_all():
    model, preprocess, device = load_clip()
    return (model, preprocess, device), {
        "Paintings": CorpusIndex("data/indices/paintings"),
    }

(model, preprocess, device), indices = load_all()

st.title("Embodied Retrieval Demo")
uploaded = st.file_uploader("Upload image",type=["png","jpg","jpeg"])
domain = st.sidebar.selectbox("Corpus", list(indices.keys()))
topk=st.sidebar.slider("Top-K",1,10,5)
alpha=st.sidebar.slider("视觉 vs 姿态 融合权重",0.0,1.0,0.8,0.05)

if uploaded:
    img=Image.open(uploaded).convert("RGB")
    st.image(img,caption="Input")
    if st.button("Find Matches"):
        qfeat=embed_image(model,preprocess,device,img)
        qpose=extract_pose(img)
        idx=indices[domain]
        sims,ids=idx.search(qfeat,topk*3)
        results=[]
        for s,i in zip(sims,ids):
            path=idx.paths[i]
            pose_s=None
            if qpose.ok and idx.poses is not None:
                cpose=idx.poses[i]
                if cpose is not None:
                    pose_s=pose_similarity(qpose.keypoints,cpose)
            final=fuse_scores(float(s),pose_s,alpha)
            results.append((path,float(s),pose_s,final))
        results.sort(key=lambda x:x[3],reverse=True)
        for path,cs,ps,fs in results[:topk]:
            cap=f"CLIP={cs:.3f} | Pose={ps if ps else 0:.3f} | Fused={fs:.3f}"
            st.image(path,caption=cap)
