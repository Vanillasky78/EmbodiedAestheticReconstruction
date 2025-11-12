# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
from PIL import Image

# 相对导入
from .config import Settings
from .model.pose_matcher import PoseMatcher

cfg = Settings()
app = FastAPI(title="EAR API", version="1.0.0")

# CORS（前端跨域时可用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局单例
matcher: Optional[PoseMatcher] = None


class MatchItem(BaseModel):
    filename: str
    title: Optional[str] = None
    artist: Optional[str] = None
    year: Optional[str] = None
    score: float


class MatchResponse(BaseModel):
    museum: str
    topk: int
    results: List[MatchItem]


@app.on_event("startup")
def _startup_load():
    """
    兼容两种 PoseMatcher 写法：
    1) 新版：PoseMatcher(data_root=..., clip_model=..., ...)
    2) 旧版：PoseMatcher(settings=Settings())
    """
    global matcher
    try:
        # 优先尝试新版签名（如果你的 pose_matcher 是我给的新版）
        matcher = PoseMatcher(
            data_root=cfg.data_root,
            clip_model=cfg.clip_model_name,
            clip_pretrained=cfg.clip_pretrained,
            device_override=cfg.device_override,
            use_dual_view=cfg.use_dual_view,
            verbose=cfg.verbose,
        )
    except TypeError:
        # 回退到旧版签名（你当前环境就是这种）
        matcher = PoseMatcher(settings=cfg)


@app.get("/status")
def status():
    if matcher is None:
        raise HTTPException(status_code=503, detail="Matcher not initialized")

    # 兼容属性名差异
    device = getattr(matcher, "device_str", None) or str(getattr(matcher, "device", "cpu"))
    museums_loaded = getattr(matcher, "museums_loaded", None)
    if museums_loaded is None and hasattr(matcher, "museums"):
        museums_loaded = list(getattr(matcher, "museums").keys())

    items_total = getattr(matcher, "items_total", None)
    if items_total is None and hasattr(matcher, "museums"):
        total = 0
        for m in matcher.museums.values():
            embs = m.get("embeddings")
            if embs is not None:
                total += int(getattr(embs, "shape", [0])[0] if hasattr(embs, "shape") else len(embs))
        items_total = total

    return {
        "museums_loaded": museums_loaded,
        "items_total": items_total,
        "device": device,
        "clip": f"{cfg.clip_model_name}/{cfg.clip_pretrained}",
    }


@app.post("/match", response_model=MatchResponse)
async def match(
    file: UploadFile = File(...),
    topk: int = Query(default=cfg.topk_default, ge=1, le=20),
    museum: Optional[str] = Query(default=None, description="e.g., local / met / tate"),
):
    if matcher is None:
        raise HTTPException(status_code=503, detail="Matcher not initialized")

    # 读取上传图像
    raw = await file.read()
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    # 兼容两种匹配 API：match_pil(...) 或 match_image_bytes(...)
    try:
        if hasattr(matcher, "match_pil"):
            results = matcher.match_pil(img, topk=topk, museum=museum)
        elif hasattr(matcher, "match_image_bytes"):
            results = matcher.match_image_bytes(raw, museum=museum, topk=topk)
        else:
            raise RuntimeError("PoseMatcher has neither match_pil nor match_image_bytes.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Matching failed: {exc}")

    # 兼容结果字段名（dict/list of objects）
    payload: List[MatchItem] = []
    for r in results:
        if isinstance(r, dict):
            filename = r.get("filename", "")
            title = r.get("title")
            artist = r.get("artist")
            year = r.get("year")
            score = float(r.get("score") or r.get("similarity") or 0.0)
        else:
            # 如果是对象，尽量取属性
            filename = getattr(r, "filename", "")
            title = getattr(r, "title", None)
            artist = getattr(r, "artist", None)
            year = getattr(r, "year", None)
            score = float(getattr(r, "score", getattr(r, "similarity", 0.0)))
        payload.append(MatchItem(filename=filename, title=title, artist=artist, year=year, score=score))

    # 默认馆名：优先入参，否则读取 matcher.default_museum
    default_museum = getattr(matcher, "default_museum", None)
    return MatchResponse(museum=museum or default_museum or "local", topk=topk, results=payload)
