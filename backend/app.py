"""Run: uvicorn backend.app:app --host 127.0.0.1 --port 8000"""
import importlib.util
import io
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from backend import tcgdex, inference

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("engine", ROOT / "dist/engine.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
app = FastAPI(title="GemLens", version="0.1.0")
Image.MAX_IMAGE_PIXELS = 24_000_000
MAX_BYTES = 15 * 1024 * 1024


@app.get("/api/health")
def health():
    available = inference.CHECKPOINT.exists() and importlib.util.find_spec('torch') is not None and importlib.util.find_spec('torchvision') is not None
    return {"status": "ok", "engine": engine.VERSION, "runtime": "python-server", "trained_model_available": available}


@app.post('/api/grade')
async def grade_pair(request: Request, front_bytes: int = Query(gt=0, le=MAX_BYTES),
                     front_crop: Optional[str] = None, back_crop: Optional[str] = None):
    """Body is original front image followed by original back image.

    Length framing avoids extra multipart dependencies and base64 expansion.
    Card boundaries are recomputed server-side, not trusted client reports.
    """
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 2 * MAX_BYTES:
            raise HTTPException(413, 'Combined photos exceed 30 MB.')
    if not 0 < len(raw)-front_bytes <= MAX_BYTES:
        raise HTTPException(422, 'Supply both front and back photos, each up to 15 MB.')
    def run():
        images, reports = [], []
        try:
            for side, data, crop in [('front', raw[:front_bytes], front_crop), ('back', raw[front_bytes:], back_crop)]:
                with Image.open(io.BytesIO(data)) as opened:
                    if opened.format not in ('JPEG', 'PNG', 'WEBP') or opened.width * opened.height > 24_000_000:
                        raise ValueError('Use JPG, PNG or WebP up to 24 megapixels.')
                    image = ImageOps.exif_transpose(opened).convert('RGB')
                small = image.convert('RGBA'); small.thumbnail((420,560))
                report = engine.analyze(small.tobytes(), *small.size, side, image.height, json.loads(crop) if crop else None)
                if not report.get('valid'):
                    raise ValueError(side.title()+': '+report.get('reason','Adjust the card boundary.'))
                if report.get('poor_capture'):
                    raise ValueError(side.title()+': Retake a sharp, evenly lit photo before grading.')
                l,t,r,b = report['bbox']
                card = image.crop((round(l*image.width),round(t*image.height),round(r*image.width),round(b*image.height)))
                if min(card.size) < 500:
                    raise ValueError(side.title()+': Use a closer, higher-resolution photo; the card must be at least 500 pixels wide.')
                images.append(card); reports.append(report)
        except (ValueError, TypeError, KeyError, UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            result = inference.predict(*images)
        except (ImportError, OSError, RuntimeError) as exc:
            raise HTTPException(503, 'The trained grader is unavailable. Check the model file and training dependencies.') from exc
        return dict(result, reports=reports)
    return await run_in_threadpool(run)


@app.post("/api/analyze")
async def inspect(request: Request, side: str = "front", crop: Optional[str] = None, original_height: int = 0):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image exceeds the 15 MB limit.")
    def run():
        try:
            im = Image.open(io.BytesIO(data))
            if im.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Use JPG, PNG or WebP.")
            if im.width * im.height > 24_000_000:
                raise ValueError("Image exceeds 24 megapixels.")
            im = ImageOps.exif_transpose(im).convert("RGBA")
            source_height = original_height or im.height
            im.thumbnail((420, 560))
            return engine.analyze(im.tobytes(), im.width, im.height, side, source_height, json.loads(crop) if crop else None)
        except (ValueError, TypeError, KeyError, UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
    return await run_in_threadpool(run)


@app.post("/api/summarize")
async def summary(request: Request):
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 100_000:
            raise HTTPException(413, "Report too large.")
    try:
        reports = json.loads(raw)
        if not isinstance(reports, list) or not 1 <= len(reports) <= 2 or any(not isinstance(r, dict) for r in reports):
            raise ValueError("Supply one or two inspection reports.")
        return engine.summarize(reports)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, "Invalid inspection reports.") from exc


# Synchronous routes run in FastAPI's thread pool; upstream I/O cannot block scans.
@app.get("/api/cards")
def search_cards(q: str = Query(min_length=2, max_length=80),
                 language: str = Query(default="en", pattern="^(en|ja|fr|de|es|it|pt|zh-tw|id|th)$"),
                 page: int = Query(default=1, ge=1, le=100)):
    if len(q.strip()) < 2:
        raise HTTPException(422, "Enter at least two characters.")
    try:
        return tcgdex.search(q.strip(), language, page)
    except tcgdex.CatalogError as exc:
        raise HTTPException(exc.status, str(exc)) from exc


@app.get("/api/cards/{card_id}")
def reference_card(card_id: str, language: str = Query(default="en", pattern="^(en|ja|fr|de|es|it|pt|zh-tw|id|th)$")):
    if len(card_id) > 100 or not all(c.isalnum() or c in "-+." for c in card_id):
        raise HTTPException(422, "Invalid card ID.")
    try:
        return tcgdex.detail(card_id, language)
    except tcgdex.CatalogError as exc:
        raise HTTPException(exc.status, str(exc)) from exc


app.mount("/", StaticFiles(directory=ROOT / "dist", html=True), name="site")
