# main.py
# Embodied Aesthetic Reconstruction — Multi-Museum FastAPI
# Endpoints:
#   GET  /                  -> health check
#   GET  /status            -> device, models, datasets
#   GET  /list_museums      -> {museum_name: count, ...}
#   GET  /metadata/{museum}/{filename}
#   POST /match             -> multipart image upload (optional: museum, topk)
#   POST /match_b64         -> base64 image (optional: museum, topk)
#   POST /batch_match       -> multiple files

from __future__ import annotations

import base64
import logging
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 按你的项目结构导入
from config import Settings
from model.pose_matcher import PoseMatcher

# ------------------------------------------------------------------------------
# App & CORS
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Embodied Aesthetic Reconstruction API (Multi-Museum)",
    description="Pose → Artwork matching across multiple open-access museums",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 如需限制来源可改为你的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
)
log = logging.getLogger("EAR-API")

# ------------------------------------------------------------------------------
# Global model holder
# ------------------------------------------------------------------------------
settings = Settings()  # 从 config.py 读取默认路径等
matcher = PoseMatcher(settings=settings)  # 加载 YOLO + CLIP + 多馆数据
log.info("EAR-API initialized: %s", matcher.get_status())

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

@app.get("/")
def root():
    """Health check."""
    return {"message": "EAR API is running"}

@app.get("/status")
def status():
    """返回设备/模型/数据集加载信息。"""
    try:
        return matcher.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_museums")
def list_museums():
    """列出所有已加载的博物馆及作品数量。"""
    try:
        return {"museums": matcher.list_museums()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metadata/{museum}/{filename}")
def metadata(museum: str, filename: str):
    """根据博物馆名与文件名返回对应元数据记录。"""
    meta = matcher.get_metadata(museum, filename)
    if not meta:
        raise HTTPException(status_code=404, detail="metadata not found")
    return meta

@app.post("/match")
async def match(
    file: UploadFile = File(...),
    museum: Optional[str] = Form(None),   # 指定单一数据源；为空则在全部数据源中检索
    topk: int = Form(3),
):
    """上传图片文件并返回 Top-K 匹配结果。"""
    try:
        img_bytes = await file.read()
        results = matcher.match_image_bytes(img_bytes, museum=museum, topk=topk)
        return JSONResponse({"results": results})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("match error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match_b64")
async def match_b64(
    image_base64: str = Form(...),        # 仅放纯 base64 字符串，不要包含 data: 前缀
    museum: Optional[str] = Form(None),
    topk: int = Form(3),
):
    """上传 base64 编码图片并返回 Top-K 匹配结果。"""
    try:
        img_bytes = base64.b64decode(image_base64)
        results = matcher.match_image_bytes(img_bytes, museum=museum, topk=topk)
        return JSONResponse({"results": results})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}")

@app.post("/batch_match")
async def batch_match(
    files: List[UploadFile] = File(...),   # 多文件上传
    museum: Optional[str] = Form(None),
    topk: int = Form(1),
):
    """批量匹配；每张图各返回一组结果。"""
    batch = []
    for f in files:
        try:
            bytes_ = await f.read()
            results = matcher.match_image_bytes(bytes_, museum=museum, topk=topk)
            batch.append({"filename": f.filename, "results": results})
        except Exception as e:
            batch.append({"filename": f.filename, "error": str(e)})
    return JSONResponse({"batch": batch})
