"""Lab state: the master dataset index in memory, the SQLite db and the single write lock.

Everything that mutates masks, taxonomy.json or the db goes through `Store.lock`
(an RLock). Frames, per-frame class pixel counts and statuses are loaded into
memory at startup; the db `classpix` rows override `meta/classpix_<split>.csv`.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import masks
from .masks import count_px, decode_counts, encode_counts, read_mask, write_mask

SPLITS = ("train", "val")
STATUSES = ("todo", "ok", "fix", "drop", "edited")
SORTS = ("time", "conf", "-conf", "low", "-edited")
MIN_FRAME_PX = 2000

SCHEMA = """
CREATE TABLE IF NOT EXISTS frames_state(stem TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'todo',
    note TEXT DEFAULT '', edited_at TEXT, edit_count INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS edits(id INTEGER PRIMARY KEY, stem TEXT, at TEXT, tool TEXT, changed_px INTEGER, delta TEXT);
CREATE TABLE IF NOT EXISTS classpix(stem TEXT PRIMARY KEY, counts TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS tax_log(id INTEGER PRIMARY KEY, at TEXT, action TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, kind TEXT, status TEXT, progress REAL, started TEXT, finished TEXT,
    params TEXT, result TEXT, log TEXT);
CREATE TABLE IF NOT EXISTS versions(id INTEGER PRIMARY KEY, at TEXT, from_version TEXT, to_version TEXT, wiped TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class FrameFilter:
    split: str | None = None
    clip: str | None = None
    status: str | None = None
    cmin: float | None = None
    cmax: float | None = None
    lmin: float | None = None
    lmax: float | None = None
    cls: int | None = None
    minpx: int = MIN_FRAME_PX
    edited: int | None = None
    sort: str = "time"


class JobHandle:
    def __init__(self, store: "Store", job_id: str):
        self.store, self.id = store, job_id
        self._log: list[str] = []
        self._last = 0.0

    def progress(self, p: float) -> None:
        t = time.monotonic()
        if t - self._last < 0.4 and p < 1.0:
            return
        self._last = t
        with self.store.lock:
            self.store.db.execute("UPDATE jobs SET progress=?, log=? WHERE id=?", (float(p), self._tail(), self.id))
            self.store.db.commit()

    def log(self, line: str) -> None:
        self._log.append(line.rstrip())
        if len(self._log) > 400:
            del self._log[:100]

    def _tail(self) -> str:
        return "\n".join(self._log)[-20000:]

    def finish(self, status: str, result: dict) -> None:
        with self.store.lock:
            self.store.db.execute(
                "UPDATE jobs SET status=?, progress=?, finished=?, result=?, log=? WHERE id=?",
                (status, 1.0 if status == "done" else None, now(), json.dumps(result), self._tail(), self.id))
            self.store.db.commit()


class Store:
    def __init__(self, root: Path, repo: Path):
        from common.taxonomy import load_taxonomy_json  # noqa: PLC0415 (path set up in lab/__init__)

        self.root, self.repo = Path(root), Path(repo)
        self.lab_dir = self.root / "lab"
        self.lab_dir.mkdir(parents=True, exist_ok=True)
        (self.lab_dir / "thumbs").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.lab_dir / "lab.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.tax_path = self.root / "taxonomy.json"
        self.tax = load_taxonomy_json(self.tax_path)
        self._load_frames()

    # ---------------------------------------------------------------- loading
    def _load_frames(self) -> None:
        with open(self.root / "frames.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["k"] = int(r["k"])
            for c in ("t_sec", "conf_mean", "conf_p10", "lowconf_frac"):
                r[c] = float(r[c])
        rows.sort(key=lambda r: (r["clip"], r["k"]))
        self.rows = rows
        n = len(rows)
        self.by_stem = {r["stem"]: i for i, r in enumerate(rows)}
        self.split_arr = np.array([r["split"] for r in rows], dtype=object)
        self.clip_arr = np.array([r["clip"] for r in rows], dtype=object)
        self.conf = np.array([r["conf_mean"] for r in rows], dtype=np.float64)
        self.low = np.array([r["lowconf_frac"] for r in rows], dtype=np.float64)
        self.status_arr = np.array(["todo"] * n, dtype=object)
        self.edit_arr = np.zeros(n, dtype=np.int64)
        self.edited_at: list[str] = [""] * n
        self.note: list[str] = [""] * n
        self.counts = np.zeros((n, 256), dtype=np.int32)

        for split in SPLITS:
            p = self.root / "meta" / f"classpix_{split}.csv"
            if not p.exists():
                continue
            with open(p, newline="") as f:
                for r in csv.DictReader(f):
                    i = self.by_stem.get(r.get("stem", ""))
                    if i is not None:
                        self.counts[i] = 0
                        for cid, px in decode_counts(r["counts"]).items():
                            self.counts[i, cid] = px
        for r in self.db.execute("SELECT stem, counts FROM classpix"):
            i = self.by_stem.get(r["stem"])
            if i is not None:
                self.counts[i] = 0
                for cid, px in decode_counts(r["counts"]).items():
                    self.counts[i, cid] = px
        for r in self.db.execute("SELECT * FROM frames_state"):
            i = self.by_stem.get(r["stem"])
            if i is not None:
                self.status_arr[i] = r["status"]
                self.edit_arr[i] = r["edit_count"] or 0
                self.edited_at[i] = r["edited_at"] or ""
                self.note[i] = r["note"] or ""

    # ------------------------------------------------------------------ paths
    def mask_path(self, stem: str) -> Path:
        return self.root / "masks" / self.rows[self.by_stem[stem]]["split"] / f"{stem}.png"

    def image_path(self, stem: str) -> Path:
        return self.root / "images" / self.rows[self.by_stem[stem]]["split"] / f"{stem}.jpg"

    def orig_path(self, stem: str) -> Path:
        return self.lab_dir / "orig" / self.rows[self.by_stem[stem]]["split"] / f"{stem}.png"

    def thumb_path(self, stem: str) -> Path:
        return self.lab_dir / "thumbs" / f"{stem}.jpg"

    # ----------------------------------------------------------------- frames
    def row_dict(self, i: int) -> dict:
        r = self.rows[i]
        return {"stem": r["stem"], "split": r["split"], "clip": r["clip"], "t_sec": r["t_sec"],
                "conf_mean": r["conf_mean"], "lowconf_frac": r["lowconf_frac"],
                "status": str(self.status_arr[i]), "edit_count": int(self.edit_arr[i])}

    def select(self, f: FrameFilter) -> list[int]:
        m = np.ones(len(self.rows), dtype=bool)
        if f.split:
            m &= self.split_arr == f.split
        if f.clip:
            m &= self.clip_arr == f.clip
        if f.status:
            m &= self.status_arr == f.status
        if f.cmin is not None:
            m &= self.conf >= f.cmin
        if f.cmax is not None:
            m &= self.conf <= f.cmax
        if f.lmin is not None:
            m &= self.low >= f.lmin
        if f.lmax is not None:
            m &= self.low <= f.lmax
        if f.cls is not None:
            m &= self.counts[:, f.cls] >= f.minpx
        if f.edited is not None:
            m &= (self.edit_arr > 0) == bool(f.edited)
        idx = np.flatnonzero(m).tolist()  # rows are already in (clip, k) order
        if f.sort == "conf":
            idx.sort(key=lambda i: self.conf[i])
        elif f.sort == "-conf":
            idx.sort(key=lambda i: -self.conf[i])
        elif f.sort == "low":
            idx.sort(key=lambda i: -self.low[i])
        elif f.sort == "-edited":
            idx.sort(key=lambda i: self.edited_at[i], reverse=True)
        return idx

    def counts_dict(self, i: int) -> dict[int, int]:
        row = self.counts[i]
        return {int(c): int(row[c]) for c in np.flatnonzero(row)}

    def class_totals(self) -> tuple[np.ndarray, np.ndarray]:
        """(frames with >= MIN_FRAME_PX px, total px) per base id (length 256)."""
        return (self.counts >= MIN_FRAME_PX).sum(axis=0), self.counts.sum(axis=0, dtype=np.int64)

    # ------------------------------------------------------------- mask edits
    def _upsert_state(self, stem: str, i: int) -> None:
        self.db.execute(
            "INSERT INTO frames_state(stem,status,note,edited_at,edit_count) VALUES(?,?,?,?,?) "
            "ON CONFLICT(stem) DO UPDATE SET status=excluded.status, note=excluded.note, "
            "edited_at=excluded.edited_at, edit_count=excluded.edit_count",
            (stem, str(self.status_arr[i]), self.note[i], self.edited_at[i] or None, int(self.edit_arr[i])))

    def _record_counts(self, stem: str, i: int, counts: np.ndarray) -> None:
        self.counts[i] = counts.astype(np.int32)
        self.db.execute(
            "INSERT INTO classpix(stem,counts,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(stem) DO UPDATE SET counts=excluded.counts, updated_at=excluded.updated_at",
            (stem, encode_counts(counts), now()))

    def apply_mask(self, stem: str, new: np.ndarray, tool: str, keep_status: bool = False) -> dict:
        with self.lock:
            i = self.by_stem[stem]
            path = self.mask_path(stem)
            old = read_mask(path)
            diff = old != new
            changed = int(diff.sum())
            if changed == 0:
                return {"changed_px": 0, "counts": self.counts_dict(i), "edit_count": int(self.edit_arr[i]),
                        "status": str(self.status_arr[i])}
            orig = self.orig_path(stem)
            if not orig.exists():
                orig.parent.mkdir(parents=True, exist_ok=True)
                tmp = orig.with_name(orig.name + ".tmp")
                shutil.copyfile(path, tmp)
                os.replace(tmp, orig)
            write_mask(path, new, self.tax.palette_bytes())
            c_new, c_old = count_px(new), count_px(old)
            delta = {str(int(c)): int(c_new[c] - c_old[c]) for c in np.flatnonzero(c_new != c_old)}
            self._record_counts(stem, i, c_new)
            self.db.execute("INSERT INTO edits(stem,at,tool,changed_px,delta) VALUES(?,?,?,?,?)",
                            (stem, now(), tool, changed, json.dumps(delta)))
            self.edit_arr[i] += 1
            self.edited_at[i] = now()
            if not keep_status and self.status_arr[i] == "todo":
                self.status_arr[i] = "edited"
            self._upsert_state(stem, i)
            self.db.commit()
            return {"changed_px": changed, "counts": self.counts_dict(i), "edit_count": int(self.edit_arr[i]),
                    "status": str(self.status_arr[i])}

    def revert(self, stem: str) -> dict:
        with self.lock:
            i = self.by_stem[stem]
            orig = self.orig_path(stem)
            if not orig.exists():
                raise FileNotFoundError(stem)
            path = self.mask_path(stem)
            cur = read_mask(path)
            os.replace(orig, path)
            back = read_mask(path)
            c_new, c_old = count_px(back), count_px(cur)
            delta = {str(int(c)): int(c_new[c] - c_old[c]) for c in np.flatnonzero(c_new != c_old)}
            self._record_counts(stem, i, c_new)
            self.db.execute("INSERT INTO edits(stem,at,tool,changed_px,delta) VALUES(?,?,?,?,?)",
                            (stem, now(), "revert", int((cur != back).sum()), json.dumps(delta)))
            self.status_arr[i] = "todo"
            self.edit_arr[i] = 0
            self.edited_at[i] = ""
            self._upsert_state(stem, i)
            self.db.commit()
            return {"counts": self.counts_dict(i), "edit_count": 0, "status": "todo"}

    def set_state(self, stem: str, status: str | None, note: str | None) -> dict:
        with self.lock:
            i = self.by_stem[stem]
            if status is not None:
                self.status_arr[i] = status
            if note is not None:
                self.note[i] = note
            self._upsert_state(stem, i)
            self.db.commit()
            return {**self.row_dict(i), "note": self.note[i]}

    # ------------------------------------------------------------------- misc
    def tax_log(self, action: str, payload: dict) -> None:
        with self.lock:
            self.db.execute("INSERT INTO tax_log(at,action,payload) VALUES(?,?,?)", (now(), action, json.dumps(payload)))
            self.db.commit()

    # ------------------------------------------------------------------- jobs
    def start_job(self, kind: str, params: dict, fn: Callable[[JobHandle], dict]) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.db.execute("INSERT INTO jobs(id,kind,status,progress,started,params,log) VALUES(?,?,?,?,?,?,?)",
                            (job_id, kind, "running", 0.0, now(), json.dumps(params), ""))
            self.db.commit()
        h = JobHandle(self, job_id)

        def run() -> None:
            try:
                result = fn(h)
                h.finish("done", result or {})
            except BaseException as e:  # noqa: BLE001 -- includes SystemExit from dataset_merge
                h.log(f"FAILED: {type(e).__name__}: {e}")
                h.finish("failed", {"error": f"{type(e).__name__}: {e}"})

        threading.Thread(target=run, name=f"job-{job_id}", daemon=True).start()
        return job_id

    def _job_row(self, r: sqlite3.Row, tail: int | None = 4000) -> dict:
        d = dict(r)
        d["params"] = json.loads(d["params"] or "null")
        d["result"] = json.loads(d["result"]) if d["result"] else None
        d["log"] = (d["log"] or "")[-tail:] if tail else ""
        return d

    def get_job(self, job_id: str) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._job_row(r) if r else None

    def list_jobs(self, kind: str | None, limit: int = 50) -> list[dict]:
        with self.lock:
            if kind:
                rs = self.db.execute("SELECT * FROM jobs WHERE kind=? ORDER BY started DESC, rowid DESC LIMIT ?", (kind, limit))
            else:
                rs = self.db.execute("SELECT * FROM jobs ORDER BY started DESC, rowid DESC LIMIT ?", (limit,))
            return [self._job_row(r, tail=400) for r in rs.fetchall()]

    def running_job(self, kind: str) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM jobs WHERE kind=? AND status='running'", (kind,)).fetchone() is not None


class ApiError(Exception):
    """Mapped to an HTTPException(status, detail) by the server."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail
