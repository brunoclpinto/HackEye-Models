"""Synthetic master dataset (real 1280x720 size) + lab app fixtures. Never touches the real master."""

from __future__ import annotations

import csv
import io
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lab import server  # noqa: E402  (also puts datasets/ on sys.path)
from common.taxonomy import data_yaml_dict, load_taxonomy_json, palette_bytes, save_merge_map  # noqa: E402

W, H = 1280, 720
ROAD, SIDEWALK, SKY, SNOW, WATER, BUILDING = 21, 24, 61, 62, 65, 27
# stem, split, clip, k, conf_mean, lowconf_frac, has_snow
FRAMES = [
    ("A_000000", "train", "A", 0, 0.95, 0.01, True),
    ("A_000001", "train", "A", 1, 0.90, 0.05, True),
    ("A_000002", "train", "A", 2, 0.80, 0.20, False),
    ("B_000000", "train", "B", 0, 0.70, 0.40, False),
    ("C_000000", "val", "C", 0, 0.85, 0.10, True),
    ("C_000001", "val", "C", 1, 0.60, 0.50, False),
]
SNOW_PX = 100 * 200


def base_mask(snow: bool) -> np.ndarray:
    m = np.full((H, W), ROAD, np.uint8)
    m[:200] = SKY
    m[400:, :300] = SIDEWALK
    m[250:300, 900:1000] = BUILDING
    if snow:
        m[100:200, 100:300] = SNOW  # inside the sky band: SNOW_PX px
    return m


def png_gray(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(arr))
    assert ok
    return buf.tobytes()


def png_rgb(arr3: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr3).save(b, format="PNG")
    return b.getvalue()


def make_master(root: Path) -> Path:
    tax = load_taxonomy_json(REPO / "taxonomies" / "vistas124.json")
    pal = palette_bytes(tax.palette)
    for d in ("images", "masks"):
        for s in ("train", "val"):
            (root / d / s).mkdir(parents=True)
    (root / "meta").mkdir()
    shutil.copy(REPO / "taxonomies" / "vistas124.json", root / "taxonomy.json")
    (root / f"{root.name}.yaml").write_text(
        yaml.safe_dump(data_yaml_dict(tax.names, tax.void_ids, f"/datasets/{root.name}"), sort_keys=False))
    rows, cp = [], {"train": [], "val": []}
    for stem, split, clip, k, conf, low, snow in FRAMES:
        m = base_mask(snow)
        im = Image.frombytes("P", (W, H), m.tobytes())
        im.putpalette(pal)
        im.save(root / "masks" / split / f"{stem}.png")
        cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), np.full((H, W, 3), 40 + k * 30, np.uint8))
        counts = np.bincount(m.ravel(), minlength=256)
        cp[split].append([stem, ";".join(f"{i}:{int(counts[i])}" for i in np.flatnonzero(counts))])
        rows.append([stem, split, clip, k, f"{k / 20:.6f}", k, 20, W, H, conf, conf - 0.1, low])
    with open(root / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("stem,split,clip,k,t_sec,src_frame,src_fps,src_w,src_h,conf_mean,conf_p10,lowconf_frac".split(","))
        w.writerows(rows)
    for split, rs in cp.items():
        with open(root / "meta" / f"classpix_{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stem", "counts"])
            w.writerows(rs)
    return root


class Env:
    def __init__(self, tmp: Path):
        self.master = make_master(tmp / "datasets" / "Master-124")
        self.exports = tmp / "exports"
        self.exports.mkdir()
        self.repo = tmp / "repo"
        (self.repo / "taxonomies").mkdir(parents=True)
        save_merge_map({"ground": ["construction--flat--road", "construction--flat--sidewalk"],
                        "wet": ["nature--snow", "nature--water"]}, self.repo / "taxonomies" / "test-map.yaml", "test map")
        self.sam_url = "http://127.0.0.1:9"  # nothing listens: SAM is down


def wait_job(client, job_id: str, timeout: float = 60.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("job timed out")


def build_client(env: Env):
    from fastapi.testclient import TestClient

    cfg = server.Config(env.master, env.exports, env.repo, env.sam_url)
    return TestClient(server.create_app(cfg))
