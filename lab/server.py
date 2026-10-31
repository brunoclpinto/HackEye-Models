"""Models Lab backend (FastAPI). See lab/CONTRACTS.md for the contract."""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__, export, masks, rules, taxonomy_api
from .versioning import check_version
from .store import SORTS, STATUSES, ApiError, FrameFilter, Store

STATIC_DIR = Path(__file__).resolve().parent / "static"
TOOLS = ("brush", "erase", "region", "polygon", "sam", "mixed")


@dataclass
class Config:
    dataset: Path
    datasets_root: Path
    repo: Path
    sam_url: str

    @staticmethod
    def from_env() -> "Config":
        return Config(Path(os.environ.get("LAB_DATASET", "/datasets/HackEye26.1-124")),
                      Path(os.environ.get("LAB_DATASETS_ROOT", "/datasets")),
                      Path(os.environ.get("LAB_REPO", "/app")),
                      os.environ.get("SAM_URL", "http://sam:8000").rstrip("/"))


class _SelectiveGZip:
    """GZip for everything except already-compressed media (png masks, jpg frames)."""

    def __init__(self, app):
        self.app = app
        self.gzip = GZipMiddleware(app, minimum_size=1024)

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "") if scope["type"] == "http" else ""
        if path.endswith((".png", ".jpg")):
            await self.app(scope, receive, send)
        else:
            await self.gzip(scope, receive, send)


class _RevalidatingStaticFiles(StaticFiles):
    """Static files are revalidated on every load (ETag keeps it cheap)."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class StateBody(BaseModel):
    status: str | None = None
    note: str | None = None


class ClassBody(BaseModel):
    name: str
    hier: str
    color: list[int]
    parent: int | None = None
    raw_name: str | None = None


class ChildBody(BaseModel):
    name: str
    color: list[int]


class SplitBody(BaseModel):
    children: list[ChildBody]


class PatchBody(BaseModel):
    name: str | None = None
    color: list[int] | None = None
    hier: str | None = None


class GroupsBody(BaseModel):
    groups: dict[str, list[str]]


class MapBody(BaseModel):
    name: str
    groups: dict[str, list[str]]
    comment: str | None = None
    overwrite: bool = False


class ExportBody(BaseModel):
    name: str
    map: str
    splits: list[str] | None = None
    groups: dict[str, list[str]] | None = None  # optional: saved as `map` when it does not exist yet


class RuleBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    from_: int = Field(alias="from")
    to: int
    filter: dict | None = None


class SamEmbedBody(BaseModel):
    stem: str


class SamPredictBody(BaseModel):
    stem: str
    points: list[list[float]] = []
    labels: list[int] = []
    box: list[float] | None = None


def create_app(config: Config | None = None) -> FastAPI:
    cfg = config or Config.from_env()
    holder: dict = {}
    init_lock = threading.Lock()

    def get_store() -> Store:
        if "store" not in holder:
            with init_lock:
                if "store" not in holder:
                    try:
                        holder["store"] = Store(cfg.dataset, cfg.repo)
                    except FileNotFoundError as e:
                        raise HTTPException(503, f"dataset not available at {cfg.dataset}: {e}") from e
        return holder["store"]

    @asynccontextmanager
    async def lifespan(_app):
        try:
            store = await run_in_threadpool(get_store)  # preload so the first request is fast
            await run_in_threadpool(check_version, store, __version__)
        except Exception:  # noqa: BLE001 -- the app still serves; requests answer 503
            pass
        yield

    app = FastAPI(title="Models Lab", version=__version__, lifespan=lifespan)
    app.add_middleware(_SelectiveGZip)
    app.state.get_store = get_store
    app.state.config = cfg

    @app.exception_handler(ApiError)
    async def _api_error(_req, exc: ApiError):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)

    def S() -> Store:
        return get_store()

    def stem_ok(store: Store, stem: str) -> int:
        i = store.by_stem.get(stem)
        if i is None:
            raise HTTPException(404, f"no frame {stem!r}")
        return i

    def frame_filter(split: str | None = None, clip: str | None = None, status: str | None = None,
                     cmin: float | None = None, cmax: float | None = None, lmin: float | None = None,
                     lmax: float | None = None, cls: int | None = None, minpx: int = 2000,
                     edited: int | None = None, sort: str = "time") -> FrameFilter:
        if sort not in SORTS:
            raise HTTPException(422, f"sort must be one of {SORTS}")
        if status and status not in STATUSES:
            raise HTTPException(422, f"status must be one of {STATUSES}")
        return FrameFilter(split, clip, status, cmin, cmax, lmin, lmax, cls, minpx, edited, sort)

    # ------------------------------------------------------------------ meta
    @app.get("/api/meta")
    def meta(store: Store = Depends(S)) -> dict:
        counts = {s: int((store.split_arr == s).sum()) for s in ("train", "val")}
        sam = {"available": False, "loaded": False}
        try:
            r = httpx.get(f"{cfg.sam_url}/health", timeout=1.0)
            if r.status_code == 200:
                sam = {"available": True, "loaded": bool(r.json().get("loaded", False))}
        except Exception:  # noqa: BLE001
            pass
        return {"version": __version__, "dataset": store.root.name, "root": str(store.root), "counts": counts, "sam": sam}

    @app.get("/api/clips")
    def clips(store: Store = Depends(S)) -> list[dict]:
        seen: dict[str, dict] = {}
        for r in store.rows:
            c = seen.setdefault(r["clip"], {"clip": r["clip"], "split": r["split"], "frames": 0})
            c["frames"] += 1
        return sorted(seen.values(), key=lambda c: c["clip"])

    # ---------------------------------------------------------------- frames
    @app.get("/api/frames")
    def frames(offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=1000),
               f: FrameFilter = Depends(frame_filter), store: Store = Depends(S)) -> dict:
        idx = store.select(f)
        return {"total": len(idx), "rows": [store.row_dict(i) for i in idx[offset:offset + limit]]}

    @app.get("/api/frames/{stem}")
    def frame(stem: str, f: FrameFilter = Depends(frame_filter), store: Store = Depends(S)) -> dict:
        i = stem_ok(store, stem)
        idx = store.select(f)
        try:
            pos = idx.index(i)
        except ValueError:
            pos = -1
        r = store.rows[i]
        return {**store.row_dict(i), "k": r["k"], "conf_p10": r["conf_p10"], "note": store.note[i],
                "edited_at": store.edited_at[i] or None,
                "prev": store.rows[idx[pos - 1]]["stem"] if pos > 0 else None,
                "next": store.rows[idx[pos + 1]]["stem"] if 0 <= pos < len(idx) - 1 else None,
                "index": pos if pos >= 0 else None, "total": len(idx)}

    @app.get("/api/frames/{stem}/image.jpg")
    def image(stem: str, store: Store = Depends(S)):
        stem_ok(store, stem)
        p = store.image_path(stem)
        if not p.exists():
            raise HTTPException(404, "image missing")
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/frames/{stem}/thumb.jpg")
    def thumb(stem: str, store: Store = Depends(S)):
        stem_ok(store, stem)
        t = store.thumb_path(stem)
        if not t.exists():
            im = cv2.imread(str(store.image_path(stem)))
            if im is None:
                raise HTTPException(404, "image missing")
            small = cv2.resize(im, (320, 180), interpolation=cv2.INTER_AREA)
            tmp = t.with_name(f"{stem}.tmp.jpg")
            cv2.imwrite(str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            os.replace(tmp, t)
        return FileResponse(t, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/frames/{stem}/mask.png")
    def mask_png(stem: str, store: Store = Depends(S)):
        stem_ok(store, stem)
        p = store.mask_path(stem)
        if not p.exists():
            raise HTTPException(404, "mask missing")
        return Response(masks.encode_gray_png(masks.read_mask(p)), media_type="image/png",
                        headers={"Cache-Control": "no-cache"})

    @app.put("/api/frames/{stem}/mask")
    async def put_mask(stem: str, request: Request, tool: str = "mixed", store: Store = Depends(S)) -> dict:
        stem_ok(store, stem)
        if tool not in TOOLS:
            raise HTTPException(422, f"tool must be one of {TOOLS}")
        body = await request.body()
        try:
            arr = masks.decode_upload(body, set(store.tax.ids))
        except masks.MaskError as e:
            raise HTTPException(422, str(e)) from e
        return await run_in_threadpool(store.apply_mask, stem, arr, tool)

    @app.post("/api/frames/{stem}/revert")
    def revert(stem: str, store: Store = Depends(S)) -> dict:
        stem_ok(store, stem)
        try:
            return store.revert(stem)
        except FileNotFoundError:
            raise HTTPException(404, "no backup for this frame") from None

    @app.put("/api/frames/{stem}/state")
    def put_state(stem: str, body: StateBody, store: Store = Depends(S)) -> dict:
        stem_ok(store, stem)
        if body.status is not None and body.status not in STATUSES:
            raise HTTPException(422, f"status must be one of {STATUSES}")
        return store.set_state(stem, body.status, body.note)

    # -------------------------------------------------------------- taxonomy
    @app.get("/api/taxonomy")
    def taxonomy(store: Store = Depends(S)) -> dict:
        return taxonomy_api.class_list(store)

    @app.post("/api/taxonomy/classes")
    def new_class(body: ClassBody, store: Store = Depends(S)) -> dict:
        return taxonomy_api.create_class(store, body.name, body.hier, body.color, body.parent, body.raw_name)

    @app.post("/api/taxonomy/classes/{cid}/split")
    def split(cid: int, body: SplitBody, store: Store = Depends(S)) -> dict:
        return taxonomy_api.split_class(store, cid, [c.model_dump() for c in body.children])

    @app.patch("/api/taxonomy/classes/{cid}")
    def patch(cid: int, body: PatchBody, store: Store = Depends(S)) -> dict:
        return taxonomy_api.patch_class(store, cid, body.name, body.color, body.hier)

    @app.delete("/api/taxonomy/classes/{cid}")
    def delete(cid: int, remap_to: int | None = None, store: Store = Depends(S)) -> dict:
        px = taxonomy_api.check_deletable(store, cid, remap_to)
        if not px:
            taxonomy_api.delete_class(store, cid)
            return {"deleted": cid}

        def job(h):
            res = rules.apply(store, cid, remap_to, None, h)
            taxonomy_api.delete_class(store, cid)
            return {**res, "deleted": cid, "remap_to": remap_to}

        return {"job": store.start_job("delete-class", {"id": cid, "remap_to": remap_to}, job)}

    # ------------------------------------------------------------------ maps
    @app.get("/api/maps")
    def maps(store: Store = Depends(S)) -> list[dict]:
        return taxonomy_api.list_maps(store)

    @app.get("/api/maps/{name}")
    def get_map(name: str, store: Store = Depends(S)) -> dict:
        return taxonomy_api.get_map(store, name)

    @app.post("/api/maps/preview")
    def map_preview(body: GroupsBody, store: Store = Depends(S)) -> dict:
        return taxonomy_api.preview(store, body.groups)

    @app.post("/api/maps")
    def save_map(body: MapBody, store: Store = Depends(S)) -> dict:
        return taxonomy_api.save_map(store, body.name, body.groups, body.comment, body.overwrite)

    # --------------------------------------------------------------- exports
    @app.post("/api/exports")
    def start_export(body: ExportBody, store: Store = Depends(S)) -> dict:
        return {"job": export.start_export(store, cfg.datasets_root, body.name, body.map, body.splits, body.groups)}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, store: Store = Depends(S)) -> dict:
        j = store.get_job(job_id)
        if j is None:
            raise HTTPException(404, "no such job")
        return j

    @app.get("/api/jobs")
    def jobs(kind: str | None = None, store: Store = Depends(S)) -> list[dict]:
        return store.list_jobs(kind)

    # ----------------------------------------------------------------- rules
    @app.post("/api/rules/dry-run")
    def rules_dry(body: RuleBody, store: Store = Depends(S)) -> dict:
        return rules.dry_run(store, body.from_, body.to, body.filter)

    @app.post("/api/rules/apply")
    def rules_apply(body: RuleBody, store: Store = Depends(S)) -> dict:
        rules.validate(store, body.from_, body.to)
        params = {"from": body.from_, "to": body.to, "filter": body.filter}
        return {"job": store.start_job("rule", params, lambda h: rules.apply(store, body.from_, body.to, body.filter, h))}

    # ------------------------------------------------------------------- sam
    def sam_call(path: str, payload: dict) -> dict:
        try:
            r = httpx.post(f"{cfg.sam_url}{path}", json=payload, timeout=120.0)
        except httpx.HTTPError as e:
            raise HTTPException(503, f"SAM service unavailable: {type(e).__name__}") from e
        if r.status_code != 200:
            raise HTTPException(502 if r.status_code >= 500 else r.status_code, f"SAM error: {r.text[:300]}")
        return r.json()

    @app.post("/api/sam/embed")
    def sam_embed(body: SamEmbedBody, store: Store = Depends(S)) -> dict:
        i = stem_ok(store, body.stem)
        return sam_call("/embed", {"stem": body.stem, "split": store.rows[i]["split"]})

    @app.post("/api/sam/predict")
    def sam_predict(body: SamPredictBody, store: Store = Depends(S)) -> dict:
        i = stem_ok(store, body.stem)
        if len(body.points) != len(body.labels) or any(len(p) != 2 for p in body.points):
            raise HTTPException(422, "points must be [[x,y],...] with one label each")
        if body.box is not None and len(body.box) != 4:
            raise HTTPException(422, "box must be [x1,y1,x2,y2]")
        return sam_call("/predict", {"stem": body.stem, "split": store.rows[i]["split"], "points": body.points,
                                     "labels": body.labels, "box": body.box})

    # ---------------------------------------------------------------- static
    @app.get("/")
    def index() -> FileResponse:
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(404, "Frontend not installed")
        return FileResponse(page, headers={"Cache-Control": "no-cache"})

    app.mount("/static", _RevalidatingStaticFiles(directory=STATIC_DIR, check_dir=False), name="static")
    return app


app = create_app()
