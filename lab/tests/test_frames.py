import io

import numpy as np
from PIL import Image

from labhelpers import (BUILDING, H, ROAD, SIDEWALK, SKY, SNOW, SNOW_PX, W, base_mask, build_client, png_gray, png_rgb,
                      wait_job)


def put(client, stem, arr, tool="brush"):
    return client.put(f"/api/frames/{stem}/mask?tool={tool}", content=png_gray(arr), headers={"content-type": "image/png"})


def test_meta_and_clips(client):
    m = client.get("/api/meta").json()
    assert m["counts"] == {"train": 4, "val": 2}
    assert m["dataset"] == "Master-124" and m["version"]
    assert m["sam"] == {"available": False, "loaded": False}
    assert client.get("/api/clips").json() == [
        {"clip": "A", "split": "train", "frames": 3}, {"clip": "B", "split": "train", "frames": 1},
        {"clip": "C", "split": "val", "frames": 2}]


def test_frames_filters_sorts_paging(client):
    g = lambda **p: client.get("/api/frames", params=p).json()  # noqa: E731
    assert g()["total"] == 6
    assert [r["stem"] for r in g()["rows"]][:2] == ["A_000000", "A_000001"]
    assert g(split="val")["total"] == 2
    assert g(clip="A")["total"] == 3
    assert g(status="ok")["total"] == 0
    assert g(cmin=0.85)["total"] == 3
    assert g(cmax=0.7)["total"] == 2
    assert g(lmin=0.4)["total"] == 2
    assert g(lmax=0.05)["total"] == 2
    assert g(cls=SNOW)["total"] == 3
    assert g(cls=SNOW, minpx=SNOW_PX + 1)["total"] == 0
    assert [r["stem"] for r in g(sort="conf")["rows"]][0] == "C_000001"
    assert [r["stem"] for r in g(sort="-conf")["rows"]][0] == "A_000000"
    assert [r["stem"] for r in g(sort="low")["rows"]][0] == "C_000001"
    assert g(offset=4, limit=10)["rows"][0]["stem"] == "C_000000"
    assert len(g(limit=2)["rows"]) == 2 and g(limit=2)["total"] == 6
    row = g()["rows"][0]
    assert set(row) == {"stem", "split", "clip", "t_sec", "conf_mean", "lowconf_frac", "status", "edit_count"}
    assert client.get("/api/frames", params={"limit": 1001}).status_code == 422
    assert client.get("/api/frames", params={"sort": "nope"}).status_code == 422
    assert client.get("/api/frames", params={"status": "nope"}).status_code == 422


def test_frame_detail_neighbours(client):
    d = client.get("/api/frames/A_000001").json()
    assert (d["prev"], d["next"], d["index"], d["total"]) == ("A_000000", "A_000002", 1, 6)
    d = client.get("/api/frames/A_000001", params={"cls": SNOW}).json()
    assert (d["prev"], d["next"], d["index"], d["total"]) == ("A_000000", "C_000000", 1, 3)
    d = client.get("/api/frames/A_000002", params={"cls": SNOW}).json()
    assert d["index"] is None and d["prev"] is None and d["total"] == 3
    assert client.get("/api/frames/nope").status_code == 404


def test_image_thumb_mask(client, env):
    r = client.get("/api/frames/A_000000/image.jpg")
    assert r.status_code == 200 and "max-age=86400" in r.headers["cache-control"]
    t = client.get("/api/frames/A_000000/thumb.jpg")
    assert Image.open(io.BytesIO(t.content)).size == (320, 180)
    assert (env.master / "lab" / "thumbs" / "A_000000.jpg").exists()
    r = client.get("/api/frames/A_000000/mask.png")
    assert r.headers["cache-control"] == "no-cache"
    im = Image.open(io.BytesIO(r.content))
    assert im.mode == "L" and im.size == (W, H)
    assert np.array_equal(np.array(im), base_mask(True))
    for p in ("image.jpg", "thumb.jpg", "mask.png"):
        assert client.get(f"/api/frames/nope/{p}").status_code == 404


def test_put_mask_roundtrip_palette_backup(client, env):
    pal = client.app.state.get_store().tax.palette
    new = base_mask(True)
    new[600:700, 500:600] = SKY  # 10000 px road/sidewalk area -> sky
    r = put(client, "A_000000", new)
    assert r.status_code == 200, r.text
    j = r.json()
    px = int((base_mask(True) != new).sum())
    assert j["changed_px"] == px and j["status"] == "edited" and j["edit_count"] == 1
    assert sum(j["counts"].values()) == W * H
    assert j["counts"][str(SKY)] == int((new == SKY).sum())
    # file on disk: P mode, ids, palette from taxonomy
    with Image.open(env.master / "masks/train/A_000000.png") as im:
        assert im.mode == "P"
        assert np.array_equal(np.array(im), new)
        p = im.getpalette()
        assert tuple(p[SKY * 3:SKY * 3 + 3]) == tuple(pal[SKY]) and tuple(p[255 * 3:255 * 3 + 3]) == (0, 0, 0)
    orig = env.master / "lab/orig/train/A_000000.png"
    assert np.array_equal(np.array(Image.open(orig)), base_mask(True))
    # second save keeps the FIRST backup
    new2 = new.copy()
    new2[0:10, 0:10] = 255
    assert put(client, "A_000000", new2).json()["edit_count"] == 2
    assert np.array_equal(np.array(Image.open(orig)), base_mask(True))
    # mask.png reflects it; cls filter reflects it
    assert np.array_equal(np.array(Image.open(io.BytesIO(client.get("/api/frames/A_000000/mask.png").content))), new2)
    assert client.get("/api/frames", params={"cls": 255, "minpx": 50}).json()["total"] == 1
    assert client.get("/api/frames", params={"edited": 1}).json()["total"] == 1
    assert client.get("/api/frames", params={"status": "edited"}).json()["rows"][0]["stem"] == "A_000000"
    # no change -> nothing written
    before = (env.master / "masks/train/A_000000.png").stat().st_mtime_ns
    r = put(client, "A_000000", new2).json()
    assert r["changed_px"] == 0 and r["edit_count"] == 2
    assert (env.master / "masks/train/A_000000.png").stat().st_mtime_ns == before
    # persisted: a fresh app on the same dataset sees counts, status and edit_count
    c2 = build_client(env)
    with c2:
        row = c2.get("/api/frames", params={"status": "edited"}).json()["rows"][0]
        assert row["stem"] == "A_000000" and row["edit_count"] == 2
        assert c2.get("/api/frames", params={"cls": 255, "minpx": 50}).json()["total"] == 1
        c2.app.state.get_store().db.close()


def test_put_mask_keeps_user_tag(client):
    assert client.put("/api/frames/A_000001/state", json={"status": "ok", "note": "fine"}).json()["note"] == "fine"
    new = base_mask(True)
    new[0:5, 0:5] = ROAD
    j = put(client, "A_000001", new).json()
    assert j["status"] == "ok" and j["edit_count"] == 1
    assert client.get("/api/frames/A_000001").json()["note"] == "fine"


def test_put_mask_422(client):
    ok = {"content-type": "image/png"}
    url = "/api/frames/A_000000/mask"
    assert client.put(url, content=png_gray(np.zeros((100, 100), np.uint8)), headers=ok).status_code == 422
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[..., 0] = 21
    rgb[..., 1] = 22
    assert client.put(url, content=png_rgb(rgb), headers=ok).status_code == 422
    bad = base_mask(True)
    bad[0, 0] = 200  # not in taxonomy
    assert put(client, "A_000000", bad).status_code == 422
    assert client.put(url, content=b"not a png", headers=ok).status_code == 422
    assert client.put(url + "?tool=nope", content=png_gray(base_mask(True)), headers=ok).status_code == 422
    assert put(client, "nope", base_mask(True)).status_code == 404
    # 255 and RGB/RGBA with R=G=B are accepted
    m = base_mask(True)
    m[0, 0] = 255
    rgba = np.stack([m, m, m, np.full_like(m, 255)], axis=-1)
    assert client.put(url, content=png_rgb(rgba), headers=ok).json()["changed_px"] == 1


def test_revert(client, env):
    assert client.post("/api/frames/A_000000/revert").status_code == 404
    new = base_mask(True)
    new[0:50, 0:50] = BUILDING
    put(client, "A_000000", new)
    r = client.post("/api/frames/A_000000/revert")
    assert r.status_code == 200 and r.json()["status"] == "todo"
    assert np.array_equal(np.array(Image.open(env.master / "masks/train/A_000000.png")), base_mask(True))
    assert not (env.master / "lab/orig/train/A_000000.png").exists()
    assert client.get("/api/frames/A_000000").json()["status"] == "todo"
    st = client.app.state.get_store()
    assert st.db.execute("SELECT tool FROM edits ORDER BY id DESC LIMIT 1").fetchone()[0] == "revert"
    assert client.post("/api/frames/A_000000/revert").status_code == 404


def test_state(client):
    j = client.put("/api/frames/A_000002/state", json={"status": "drop"}).json()
    assert j["status"] == "drop"
    assert client.get("/api/frames", params={"status": "drop"}).json()["total"] == 1
    assert client.put("/api/frames/A_000002/state", json={"status": "bogus"}).status_code == 422
    assert client.put("/api/frames/nope/state", json={"status": "ok"}).status_code == 404
