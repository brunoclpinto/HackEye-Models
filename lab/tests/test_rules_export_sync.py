import importlib.util
import os
import sqlite3
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from labhelpers import REPO, ROAD, SIDEWALK, SNOW, SNOW_PX, WATER, base_mask, png_gray, wait_job


def test_rules_dry_run_vs_apply(client, env):
    client.put("/api/frames/A_000000/state", json={"status": "fix"})
    d = client.post("/api/rules/dry-run", json={"from": SNOW, "to": WATER, "filter": {}}).json()
    assert d["frames"] == 3 and d["pixels"] == 3 * SNOW_PX and len(d["sample"]) == 3
    assert client.post("/api/rules/dry-run", json={"from": SNOW, "to": WATER, "filter": {"split": "val"}}).json()["frames"] == 1
    assert client.post("/api/rules/dry-run", json={"from": SNOW, "to": WATER, "filter": {"clip": "A"}}).json()["frames"] == 2
    assert client.post("/api/rules/dry-run", json={"from": SNOW, "to": WATER, "filter": {"stems": ["A_000000"]}}).json()["frames"] == 1
    assert client.post("/api/rules/dry-run", json={"from": SNOW, "to": SNOW}).status_code == 422
    assert client.post("/api/rules/dry-run", json={"from": SNOW, "to": 200}).status_code == 422
    assert client.post("/api/rules/dry-run", json={"from": 124, "to": 3}).status_code == 422
    # dry run changed nothing
    assert np.array_equal(np.array(Image.open(env.master / "masks/train/A_000000.png")), base_mask(True))
    j = wait_job(client, client.post("/api/rules/apply", json={"from": SNOW, "to": WATER, "filter": {"split": "train"}}).json()["job"])
    assert j["status"] == "done" and j["result"] == {"frames": 2, "pixels": 2 * SNOW_PX}, j
    got = np.array(Image.open(env.master / "masks/train/A_000000.png"))
    assert not (got == SNOW).any() and (got == WATER).sum() == SNOW_PX
    assert np.array(Image.open(env.master / "lab/orig/train/A_000000.png")).max() >= SNOW
    # statuses kept (fix stays, todo stays todo), edit counted
    rows = {r["stem"]: r for r in client.get("/api/frames").json()["rows"]}
    assert rows["A_000000"]["status"] == "fix" and rows["A_000001"]["status"] == "todo"
    assert rows["A_000000"]["edit_count"] == 1 and rows["C_000000"]["edit_count"] == 0
    tool = client.app.state.get_store().db.execute("SELECT DISTINCT tool FROM edits").fetchall()
    assert [t[0] for t in tool] == [f"rule:{SNOW}->{WATER}"]
    assert client.post("/api/rules/dry-run", json={"from": SNOW, "to": WATER}).json()["frames"] == 1
    assert client.get("/api/jobs", params={"kind": "rule"}).json()[0]["id"] == j["id"]
    assert client.get("/api/jobs/nope").status_code == 404


def test_export_end_to_end(client, env):
    new = base_mask(True)
    new[300:310, 600:610] = WATER
    client.put("/api/frames/C_000000/mask", content=png_gray(new), headers={"content-type": "image/png"})
    assert client.post("/api/exports", json={"name": "bad name", "map": "test-map"}).status_code == 422
    assert client.post("/api/exports", json={"name": "Exp-1", "map": "nope"}).status_code == 404
    r = client.post("/api/exports", json={"name": "Exp-1", "map": "test-map"})
    assert r.status_code == 200, r.text
    j = wait_job(client, r.json()["job"])
    assert j["status"] == "done", j
    assert j["result"]["ok"] and j["result"]["n_classes"] == 122 and j["progress"] == 1.0
    assert all(v["errors"] == [] and v["accounting_mismatches"] == [] for v in j["result"]["verify"].values())
    assert "pixel accounting OK" in j["log"]
    dst = env.exports / "Exp-1"
    y = yaml.safe_load((dst / "Exp-1.yaml").read_text())
    assert len(y["names"]) == 122 and "wet" in y["names"].values()
    # hardlinked images, remapped masks
    for stem, split in (("A_000000", "train"), ("C_000000", "val")):
        a, b = env.master / "images" / split / f"{stem}.jpg", dst / "images" / split / f"{stem}.jpg"
        assert os.path.samestat(a.stat(), b.stat()) and a.stat().st_nlink >= 2
    out = np.array(Image.open(dst / "masks/val/C_000000.png"))
    lut = client.post("/api/maps/preview", json={"groups": client.get("/api/maps/test-map").json()["groups"]}).json()["lut"]
    assert np.array_equal(out, np.array(lut, np.uint8)[new])  # includes the edit
    assert len(list((dst / "masks" / "train").glob("*.png"))) == 4
    # exists -> 409; export with groups creates the map; ignore group works
    assert client.post("/api/exports", json={"name": "Exp-1", "map": "test-map"}).status_code == 409
    r = client.post("/api/exports", json={"name": "Exp-2", "map": "new-map", "splits": ["val"],
                                          "groups": {"__ignore__": ["nature--snow"]}})
    assert r.status_code == 200
    j2 = wait_job(client, r.json()["job"])
    assert j2["status"] == "done", j2
    assert (env.repo / "taxonomies" / "new-map.yaml").exists()
    o2 = np.array(Image.open(env.exports / "Exp-2/masks/val/C_000000.png"))
    assert (o2 == 255).sum() == SNOW_PX
    assert not (env.exports / "Exp-2/masks/train").exists() or not list((env.exports / "Exp-2/masks/train").glob("*"))


def test_export_failure_recorded(client, env):
    (env.master / "images/train/A_000000.jpg").unlink()
    j = wait_job(client, client.post("/api/exports", json={"name": "Exp-F", "map": "test-map"}).json()["job"])
    assert j["status"] == "failed" and "error" in j["result"]


def test_sam_proxy(client, monkeypatch):
    assert client.post("/api/sam/embed", json={"stem": "A_000000"}).status_code == 503
    assert client.post("/api/sam/embed", json={"stem": "nope"}).status_code == 404
    from lab import server

    seen = {}

    class R:
        status_code = 200
        text = ""

        def json(self):
            return {"masks": [{"score": 0.9, "png_b64": "xx"}]} if "predict" in seen["url"] else {"ok": True}

    def fake_post(url, json=None, timeout=None):
        seen.update(url=url, json=json)
        return R()

    monkeypatch.setattr(server.httpx, "post", fake_post)
    r = client.post("/api/sam/predict", json={"stem": "C_000000", "points": [[1, 2]], "labels": [1], "box": None})
    assert r.json()["masks"][0]["score"] == 0.9
    assert seen["json"] == {"stem": "C_000000", "split": "val", "points": [[1.0, 2.0]], "labels": [1], "box": None}
    assert client.post("/api/sam/predict", json={"stem": "C_000000", "points": [[1, 2]], "labels": []}).status_code == 422
    assert client.post("/api/sam/embed", json={"stem": "A_000000"}).json() == {"ok": True}
    assert seen["json"] == {"stem": "A_000000", "split": "train"}


def test_static_and_index(client):
    assert client.get("/static/none.js").status_code == 404
    assert client.get("/").status_code in (200, 404)  # 404 until lab/static/index.html exists


def test_missing_dataset_is_503_not_import_error(tmp_path):
    from fastapi.testclient import TestClient
    from lab import server

    c = TestClient(server.create_app(server.Config(tmp_path / "none", tmp_path, tmp_path, "http://127.0.0.1:9")))
    with c:
        assert c.get("/api/meta").status_code == 503


def test_sync_lab_edits(client, env):
    spec = importlib.util.spec_from_file_location("sync_lab_edits", REPO / "datasets/hackeye/sync_lab_edits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.sync(env.master) == {"train": 0, "val": 0}  # no edits yet
    new = base_mask(True)
    new[300:310, 600:610] = WATER
    client.put("/api/frames/C_000000/mask", content=png_gray(new), headers={"content-type": "image/png"})
    client.app.state.get_store().db.execute("PRAGMA wal_checkpoint(FULL)")
    assert mod.sync(env.master, dry_run=True) == {"train": 0, "val": 1}
    assert mod.sync(env.master) == {"train": 0, "val": 1}
    rows = (env.master / "meta/classpix_val.csv").read_text().splitlines()
    assert len(rows) == 3 and rows[0] == "stem,counts"
    c = dict(p.split(":") for p in dict(r.split(",") for r in rows[1:])["C_000000"].split(";"))
    assert int(c[str(WATER)]) == 100 and int(c[str(SNOW)]) == SNOW_PX
    assert mod.sync(env.master) == {"train": 0, "val": 0}  # idempotent
