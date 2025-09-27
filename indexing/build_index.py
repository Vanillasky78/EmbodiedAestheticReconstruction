# indexing/build_index.py
import os, json, faiss, numpy as np, tqdm
from PIL import Image
from processors.pose import extract_pose
from processors.retrieval import load_clip, embed_image

def iter_images(root_dir):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(("jpg","jpeg","png")):
                yield os.path.join(root, f)

def build(corpus, outdir, max_images=0):
    os.makedirs(outdir, exist_ok=True)
    model, preprocess, device = load_clip()
    feats, paths, poses = [], [], []
    for i, path in enumerate(tqdm.tqdm(iter_images(corpus))):
        if max_images and i>=max_images: break
        try:
            img = Image.open(path).convert("RGB")
            feats.append(embed_image(model, preprocess, device, img)[0])
            pres = extract_pose(img)
            poses.append(pres.keypoints if pres.ok else None)
            paths.append({"path": path})
        except: continue
    feats = np.stack(feats).astype("float32")
    faiss.normalize_L2(feats)
    index = faiss.index_factory(feats.shape[1], "IDMap,Flat")
    index.add_with_ids(feats, np.arange(len(feats)))
    np.save(os.path.join(outdir,"feats.f32.npy"),feats)
    np.save(os.path.join(outdir,"poses.npy"),np.array(poses,dtype=object))
    with open(os.path.join(outdir,"paths.jsonl"),"w",encoding="utf-8") as f:
        for rec in paths: f.write(json.dumps(rec)+"\n")
    faiss.write_index(index, os.path.join(outdir,"faiss.index"))
    print("saved",len(paths),"images")

if __name__=="__main__":
    build("data/corpus_sample/paintings","data/indices/paintings",max_images=50)
