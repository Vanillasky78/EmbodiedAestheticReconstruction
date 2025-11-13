"""
EAR Backend API (FastAPI + Uvicorn)

Exposes:
- GET  /health     → quick status check
- POST /match      → upload an image and get Top-K matched artworks

It uses the PoseMatcher defined in backend/model/pose_matcher.py and the
Settings from backend/config.py.
"""

from __future__ import annotations

import io
import logging
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from .config import Settings
from .model.pose_matcher import PoseMatcher


logger = logging.getLogger("uvicorn.error")

# ----------------------------------------------------------------------
# FastAPI app + CORS
# ----------------------------------------------------------------------

app = FastAPI(
    title="Embodied Aesthetic Reconstruction — Backend API",
    version="0.1.0",
    description="CLIP-based portrait → artwork matching backend.",
)

# Allow everything locally; you can tighten this later if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global objects filled at startup
settings: Optional[Settings] = None
matcher: Optional[PoseMatcher] = None


# ----------------------------------------------------------------------
# Startup / shutdown
# ----------------------------------------------------------------------


@app.on_event("startup")
def _startup_load() -> None:
    """Load configuration + matcher once when the server starts."""
    global settings, matcher

    logger.info("Loading Settings and PoseMatcher…")
    settings = Settings()
    matcher = PoseMatcher(settings)
    logger.info("Matcher loaded successfully.")


@app.on_event("shutdown")
def _shutdown_cleanup() -> None:
    """Optional cleanup (currently nothing special)."""
    global matcher
    matcher = None
    logger.info("Matcher released.")


# ----------------------------------------------------------------------
# Simple health endpoint
# ----------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": getattr(matcher, "device", "unknown") if matcher else "unloaded",
        "default_museum": getattr(settings, "default_museum", None)
        if settings
        else None,
    }


# ----------------------------------------------------------------------
# Schemas for /match
# ----------------------------------------------------------------------


class MatchResult(BaseModel):
    filename: str
    title: Optional[str] = None
    artist: Optional[str] = None
    year: Optional[int] = None
    score: float


class MatchResponse(BaseModel):
    museum: Optional[str]
    topk: int
    results: List[MatchResult]


# ----------------------------------------------------------------------
# Main matching endpoint
# ----------------------------------------------------------------------


@app.post("/match", response_model=MatchResponse)
async def match(
    image: UploadFile = File(..., description="Uploaded portrait photo"),
    museum: Optional[str] = Form(None, description="Museum name, e.g. local / met"),
    topk: int = Form(3, description="Number of top results"),
):
    """
    Robust matching endpoint:

    - validates that the uploaded file looks like an image
    - decodes it to a PIL image (RGB)
    - calls PoseMatcher.match_pil(...)
    - tries to back-fill missing title / artist / year from matcher.meta_by_filename
    """
    # ----------------- basic validation -----------------
    if topk <= 0:
        topk = 3

    if image.content_type is not None and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {image.content_type}. Please upload an image file.",
        )

    data = await image.read()
    if not data:
        raise HTTPException(
            status_code=400,
            detail="Empty file received. Please take a photo again.",
        )

    # ----------------- decode to PIL -----------------
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        logger.exception("Failed to decode uploaded image")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {type(exc).__name__}",
        )

    # ----------------- check matcher -----------------
    global matcher
    if matcher is None:
        raise HTTPException(
            status_code=500,
            detail="Matcher is not loaded on the server.",
        )

    effective_museum = museum or getattr(matcher, "default_museum", None) or "local"

    # ----------------- run matcher -----------------
    try:
        if hasattr(matcher, "match_pil"):
            raw_results = matcher.match_pil(
                pil_img,
                museum=effective_museum,
                topk=topk,
            )
        else:
            raise RuntimeError("PoseMatcher has no method 'match_pil'")
    except ValueError as exc:
        logger.exception("ValueError in matcher.match_pil")
        raise HTTPException(
            status_code=400,
            detail=f"Matching failed: {exc}",
        )
    except Exception as exc:
        logger.exception("Unhandled error in matcher.match_pil")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during matching: {type(exc).__name__}: {exc}",
        )

    # ----------------- normalize results structure -----------------
    payload: List[MatchResult] = []

    meta_store = getattr(matcher, "meta_by_filename", None)

    def fill_from_meta(filename: str, title, artist, year):
        """If title/artist/year are missing, try to fill from matcher metadata."""
        nonlocal meta_store
        if not isinstance(meta_store, dict):
            return title, artist, year

        meta = meta_store.get(filename) or meta_store.get(filename.strip()) or {}
        if isinstance(meta, dict):
            title = title or meta.get("title") or meta.get("Title")
            artist = artist or meta.get("artist") or meta.get("Artist")
            year = year or meta.get("year") or meta.get("Year")
        else:
            title = title or getattr(meta, "title", None)
            artist = artist or getattr(meta, "artist", None)
            year = year or getattr(meta, "year", None)
        return title, artist, year

    for r in raw_results or []:
        if isinstance(r, dict):
            filename = r.get("filename") or r.get("file") or ""
            score = float(r.get("score", 0.0))
            title = r.get("title")
            artist = r.get("artist")
            year = r.get("year")
        else:
            filename = getattr(r, "filename", "") or getattr(r, "file", "")
            score = float(getattr(r, "score", 0.0))
            title = getattr(r, "title", None)
            artist = getattr(r, "artist", None)
            year = getattr(r, "year", None)

        title, artist, year = fill_from_meta(filename, title, artist, year)

        # normalise year → int or None
        year_int: Optional[int]
        try:
            if isinstance(year, str) and year.strip():
                year_int = int(year.split(",")[0].strip())
            elif isinstance(year, (int, float)):
                year_int = int(year)
            else:
                year_int = None
        except Exception:
            year_int = None

        payload.append(
            MatchResult(
                filename=filename,
                title=title,
                artist=artist,
                year=year_int,
                score=float(score),
            )
        )

    return MatchResponse(
        museum=effective_museum,
        topk=topk,
        results=payload,
    )
