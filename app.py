import streamlit as st, json
from PIL import Image
from processors.retrieval import Matcher

st.sidebar.header("Match-Only Settings")
w_clip = st.sidebar.slider("Weight: CLIP", 0.0, 1.0, 0.6, 0.05)
w_pose = st.sidebar.slider("Weight: Pose", 0.0, 1.0, 0.3, 0.05)
w_color = st.sidebar.slider("Weight: Color", 0.0, 1.0, 0.1, 0.05)
require_pd = st.sidebar.checkbox("Require Public Domain/CC0", value=True)
TOPK = st.sidebar.slider("Top-K", 5, 50, 12)

# assume `matcher` is instantiated during app init
img_file = st.file_uploader("Upload a portrait photo", type=["jpg","jpeg","png"]) 
if img_file:
    q = Image.open(img_file)
    st.image(q, caption="Input", use_column_width=True)
    weights = dict(w_clip=w_clip, w_pose=w_pose, w_color=w_color)
    results = matcher.search(q, k=200, weights=weights, filters={"require_public_domain": require_pd})
    results = results[:TOPK]

    cols = st.columns(min(4, TOPK))
    for i, (rid, meta, sc) in enumerate(results):
        c = cols[i % len(cols)]
        with c:
            st.markdown(f"**{meta.get('artwork_title_en','Untitled')}**")
            st.caption(f"{meta.get('artist_name_en','Unknown')} • {meta.get('year','?')} • {meta.get('museum','')} ")
            st.caption(f"License: {meta.get('license','?')}  |  Score: {sc:.3f}")
            if meta.get('image_path'):
                st.image(meta['image_path'])