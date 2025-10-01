# indexing/build_index.py
import os, json, faiss, argparse, numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from processors.pose import extract_pose
from processors.retrieval import load_clip, embed_image

def iter_images(root_dir):
    exts = (".jpg",".jpeg",".png",".webp",".bmp")
    for root,_,files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                yield os.path.join(root,f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--tag", action="append", default=[], help="key=value, e.g. --tag domain=fashion --tag brand=Balenciaga")
    args = ap.parse_args()

    tags = {}
    for kv in args.tag:
        if "=" in kv: k,v=kv.split("=",1); tags[k]=v

    os.makedirs(args.outdir, exist_ok=True)
    model, preprocess, device = load_clip()

    feats, paths, poses = [], [], []
    imgs = list(iter_images(args.corpus))
    if args.max_images>0: imgs=imgs[:args.max_images]

    for path in tqdm(imgs,desc="Embedding+Pose"):
        try:
            img = Image.open(path).convert("RGB")
            f = embed_image(model,preprocess,device,img)[0]
            feats.append(f)
            pres = extract_pose(img)
            poses.append(pres.keypoints if pres.ok else None)
            rec={"path":path,"tags":tags}
            paths.append(rec)
        except Exception as e:
            print("skip:",path,e)

    feats=np.stack(feats).astype("float32"); faiss.normalize_L2(feats)
    index=faiss.index_factory(feats.shape[1],"IDMap,Flat")
    index.add_with_ids(feats,np.arange(len(feats)).astype("int64"))

    np.save(os.path.join(args.outdir,"feats.f32.npy"),feats)
    np.save(os.path.join(args.outdir,"poses.npy"),np.array(poses,dtype=object))
    with open(os.path.join(args.outdir,"paths.jsonl"),"w",encoding="utf-8") as f:
        for rec in paths: f.write(json.dumps(rec,ensure_ascii=False)+"\n")
    faiss.write_index(index, os.path.join(args.outdir,"faiss.index"))
    print(f"[done] saved index to {args.outdir} with {len(paths)} images")

if __name__=="__main__":
    main()
