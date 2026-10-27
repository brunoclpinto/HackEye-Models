"""Bulk class rules: relabel every pixel of class `from` to `to` across a frame selection."""

from __future__ import annotations

import numpy as np

from .masks import read_mask
from .store import ApiError, JobHandle, Store


def validate(store: Store, src: int, dst: int) -> None:
    if src not in store.tax.ids:
        raise ApiError(422, f"from {src} is not a class")
    if dst != 255 and dst not in store.tax.ids:
        raise ApiError(422, f"to {dst} is not a class")
    if src == dst:
        raise ApiError(422, "from and to are the same class")


def select(store: Store, src: int, flt: dict | None) -> list[int]:
    flt = flt or {}
    m = store.counts[:, src] > 0
    if flt.get("split"):
        m &= store.split_arr == flt["split"]
    if flt.get("clip"):
        m &= store.clip_arr == flt["clip"]
    if flt.get("stems") is not None:
        wanted = set(flt["stems"])
        m &= np.array([r["stem"] in wanted for r in store.rows], dtype=bool)
    return np.flatnonzero(m).tolist()


def dry_run(store: Store, src: int, dst: int, flt: dict | None) -> dict:
    validate(store, src, dst)
    idx = select(store, src, flt)
    return {"frames": len(idx), "pixels": int(store.counts[idx, src].sum()) if idx else 0,
            "sample": [store.rows[i]["stem"] for i in idx[:20]]}


def apply(store: Store, src: int, dst: int, flt: dict | None, h: JobHandle | None = None) -> dict:
    validate(store, src, dst)
    idx = select(store, src, flt)
    tool = f"rule:{src}->{dst}"
    frames = changed = 0
    for n, i in enumerate(idx, 1):
        stem = store.rows[i]["stem"]
        arr = read_mask(store.mask_path(stem))
        new = arr.copy()
        new[arr == src] = dst
        res = store.apply_mask(stem, new, tool, keep_status=True)
        if res["changed_px"]:
            frames += 1
            changed += res["changed_px"]
        if h:
            h.progress(n / len(idx))
    if h:
        h.log(f"{tool}: {frames} frames, {changed} px")
    return {"frames": frames, "pixels": changed}
