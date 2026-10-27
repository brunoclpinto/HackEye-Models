"""Export job: derives a merged dataset (datasets/common/dataset_merge.py) from the master."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from common import dataset_merge
from common.taxonomy import load_merge_map

from . import taxonomy_api
from .store import ApiError, JobHandle, Store

NAME_RE = taxonomy_api.NAME_RE
WORKERS = 4


def start_export(store: Store, datasets_root: Path, name: str, map_name: str, splits: list[str] | None,
                 groups: dict | None = None) -> str:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ApiError(422, "export name must match ^[A-Za-z0-9][A-Za-z0-9._-]*$")
    splits = splits or ["train", "val"]
    if not splits or any(s not in ("train", "val") for s in splits):
        raise ApiError(422, "splits must be a subset of [train, val]")
    dst = Path(datasets_root) / name
    if dst.exists():
        raise ApiError(409, f"{dst} exists")
    mpath = taxonomy_api.map_path(store.repo, map_name)
    if not mpath.exists():
        if groups is None:
            raise ApiError(404, f"no map {map_name!r}")
        taxonomy_api.save_map(store, map_name, groups, f"saved by the Lab with export {name}", False)
    try:
        load_merge_map(mpath)
    except ValueError as e:
        raise ApiError(422, str(e)) from e
    if store.running_job("export"):
        raise ApiError(409, "another export is running")

    tax_sha = hashlib.sha256(store.tax_path.read_bytes()).hexdigest()

    def job(h: JobHandle) -> dict:
        orig_log = dataset_merge._log
        dataset_merge._log = lambda msg: h.log(f"{dataset_merge.TAG} {msg}")

        def progress(split, i, total):
            h.progress((splits.index(split) + i / max(total, 1)) / len(splits))

        try:
            ok = dataset_merge.run(store.root, dst, mpath, store.tax, tax_sha, splits, WORKERS, False,
                                   progress=progress)
        finally:
            dataset_merge._log = orig_log
        man = json.loads((dst / "merge_manifest.json").read_text()) if (dst / "merge_manifest.json").exists() else {}
        verify = {s: {"merged": v.get("merged"), "errors": v.get("errors"),
                      "accounting_mismatches": v.get("accounting_mismatches")}
                  for s, v in man.get("splits", {}).items()}
        res = {"ok": bool(ok), "dst": str(dst), "yaml": str(dst / f"{name}.yaml"), "map": map_name,
               "n_classes": man.get("n_classes"), "verify": verify}
        if not ok:
            raise RuntimeError("verification failed: " + json.dumps(verify)[:2000])
        return res

    return store.start_job("export", {"name": name, "map": map_name, "splits": splits}, job)
