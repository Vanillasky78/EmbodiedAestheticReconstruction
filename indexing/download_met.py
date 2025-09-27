# indexing/download_met.py
import os, requests
from PIL import Image
from io import BytesIO

SAVE_DIR="data/corpus_sample/paintings"
N=50
os.makedirs(SAVE_DIR,exist_ok=True)

r=requests.get("https://collectionapi.metmuseum.org/public/collection/v1/search",
               params={"q":"painting","hasImages":"true","isPublicDomain":"true"})
ids=r.json()["objectIDs"][:N]

for oid in ids:
    url=f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}"
    info=requests.get(url).json()
    img_url=info.get("primaryImageSmall")
    if not img_url: continue
    try:
        img=Image.open(BytesIO(requests.get(img_url).content)).convert("RGB")
        img.save(os.path.join(SAVE_DIR,f"{oid}.jpg"))
    except: pass

print("done, saved to",SAVE_DIR)
