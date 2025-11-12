# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
from PIL import Image

# ✅ 相对导入（关键改动）
from .config import Settings
from .model.pose_matcher import PoseMatcher

cfg = Settings()
app = FastAPI(title="EAR API", version="1.0.0")

# CORS（需要前端跨域时可保留）
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
    """Load models and embeddings once on server start."""
    global matcher
    matcher = PoseMatcher(
        data_root=cfg.data_root,
        clip_model=cfg.clip_model_name,
        clip_pretrained=cfg.clip_pretrained,
        device_override=cfg.device_override,
        use_dual_view=cfg.use_dual_view,
        verbose=cfg.verbose,
    )


@app.get("/status")
def status():
    if matcher is None:
        raise HTTPException(status_code=503, detail="Matcher not initialized")
    return {
        "museums_loaded": matcher.museums_loaded,  # e.g. ["local"]
        "items_total": matcher.items_total,
        "device": matcher.device_str,
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
    try:
        raw = await file.read()
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    # 执行匹配
    try:
        results = matcher.match_pil(img, topk=topk, museum=museum)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Matching failed: {exc}")

    payload = [
        MatchItem(
            filename=r["filename"],
            title=r.get("title"),
            artist=r.get("artist"),
            year=r.get("year"),
            score=float(r["score"]),
        )
        for r in results
    ]
    return MatchResponse(
        museum=museum or matcher.default_museum,
        topk=topk,
        results=payload,
    )
