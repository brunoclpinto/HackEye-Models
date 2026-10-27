"""SAM2 click/box segmentation service for the Models Lab (lab/PLAN.md, "SAM service").

Lazy model load, LRU(8) of image embeddings keyed by stem, idle unload after SAM_IDLE_S, one request at a time.
"""
import base64
import io
import os
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "/models/sam2.1_hiera_large.pt")
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DATASET = os.environ.get("SAM_DATASET", "HackEye26.1-124")
IDLE_S = float(os.environ.get("SAM_IDLE_S", "600"))
CACHE_N = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI(title="Models Lab SAM2")

_lock = threading.Lock()
_predictor = None
_cache: "OrderedDict[str, dict]" = OrderedDict()   # stem -> {"features", "orig_hw"}
_last_used = time.time()
_current = None                                    # stem whose features are set on the predictor


def _load():
    global _predictor
    if _predictor is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _predictor = SAM2ImagePredictor(build_sam2(CONFIG, CHECKPOINT, device=device))
    return _predictor


def _unload():
    global _predictor, _current
    _predictor = None
    _current = None
    _cache.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _idle_watch():
    while True:
        time.sleep(min(5.0, max(1.0, IDLE_S / 4)))
        if _predictor is not None and time.time() - _last_used > IDLE_S:
            if _lock.acquire(blocking=False):
                try:
                    if _predictor is not None and time.time() - _last_used > IDLE_S:
                        _unload()
                finally:
                    _lock.release()


threading.Thread(target=_idle_watch, daemon=True).start()


def _embed(stem, split):
    """Make the predictor hold the features of `stem` (computing and caching them if needed). Lock must be held."""
    global _current
    pred = _load()
    if stem in _cache:
        _cache.move_to_end(stem)
        if _current != stem:
            e = _cache[stem]
            pred._features, pred._orig_hw, pred._is_image_set = e["features"], e["orig_hw"], True
            _current = stem
        return False
    path = f"/datasets/{DATASET}/images/{split}/{stem}.jpg"
    if not os.path.isfile(path):
        raise HTTPException(404, f"image not found: {path}")
    arr = np.array(Image.open(path).convert("RGB"))
    pred.set_image(arr)
    if torch.cuda.is_available():
        torch.cuda.synchronize()     # honest embed timing
    _cache[stem] = {"features": pred._features, "orig_hw": pred._orig_hw}
    _current = stem
    while len(_cache) > CACHE_N:
        _cache.popitem(last=False)
    return True


def _png_b64(mask):
    buf = io.BytesIO()
    Image.fromarray((mask > 0).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class EmbedReq(BaseModel):
    stem: str
    split: str


class PredictReq(BaseModel):
    stem: str
    split: str
    points: list[list[float]] = []
    labels: list[int] = []
    box: list[float] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "loaded": _predictor is not None, "cached": len(_cache)}


@app.post("/embed")
def embed(req: EmbedReq):
    global _last_used
    with _lock:
        t = time.time()
        fresh = _embed(req.stem, req.split)
        _last_used = time.time()
        return {"stem": req.stem, "cached": not fresh, "ms": round((time.time() - t) * 1000)}


@app.post("/predict")
def predict(req: PredictReq):
    global _last_used
    if not req.points and req.box is None:
        raise HTTPException(422, "need points or box")
    if len(req.labels) != len(req.points):
        raise HTTPException(422, "labels and points differ in length")
    with _lock:
        _embed(req.stem, req.split)
        pred = _predictor
        pts = np.array(req.points, dtype=np.float32) if req.points else None
        lab = np.array(req.labels, dtype=np.int32) if req.points else None
        box = np.array(req.box, dtype=np.float32) if req.box else None
        with torch.inference_mode():
            masks, scores, _ = pred.predict(point_coords=pts, point_labels=lab, box=box, multimask_output=True)
        _last_used = time.time()
        order = np.argsort(-scores)[:3]
        return {"masks": [{"score": float(scores[i]), "png_b64": _png_b64(masks[i])} for i in order]}
