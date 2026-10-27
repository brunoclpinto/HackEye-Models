import json

import numpy as np
import yaml
from PIL import Image

from labhelpers import ROAD, SIDEWALK, SNOW, SNOW_PX, WATER, W, H, base_mask, png_gray, wait_job


def by_id(client, cid):
    return next(c for c in client.get("/api/taxonomy").json()["classes"] if c["id"] == cid)


def test_taxonomy_list(client):
    cl = client.get("/api/taxonomy").json()["classes"]
    assert len(cl) == 124 and cl[0]["id"] == 0
    snow = by_id(client, SNOW)
    assert snow["frames"] == 3 and snow["pixels"] == 3 * SNOW_PX and snow["custom"] is False
    assert set(snow) == {"id", "name", "raw_name", "hier", "color", "void", "parent", "custom", "frames", "pixels"}


def test_create_class(client, env):
    r = client.post("/api/taxonomy/classes", json={"name": "curb ramp", "hier": "flat", "color": [1, 2, 3]})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["id"] == 124 and c["raw_name"] == "hackeye--flat--curb-ramp" and c["parent"] is None
    assert by_id(client, 124)["custom"] is True
    tj = json.loads((env.master / "taxonomy.json").read_text())
    assert tj["classes"][-1]["id"] == 124
    assert (env.master / "taxonomy.json.bak").exists()
    y = yaml.safe_load((env.master / "Master-124.yaml").read_text())
    assert y["names"][124] == "curb ramp" and len(y["names"]) == 125
    assert (env.master / "Master-124.yaml.bak").exists()
    log = client.app.state.get_store().db.execute("SELECT action FROM tax_log").fetchall()
    assert [r[0] for r in log] == ["create"]
    # duplicates and bad input
    assert client.post("/api/taxonomy/classes", json={"name": "curb ramp", "hier": "flat", "color": [1, 2, 3]}).status_code == 409
    assert client.post("/api/taxonomy/classes", json={"name": "other", "hier": "flat", "color": [1, 2, 3],
                                                      "raw_name": "hackeye--flat--curb-ramp"}).status_code == 409
    assert client.post("/api/taxonomy/classes", json={"name": "x", "hier": "flat", "color": [1, 2, 999]}).status_code == 422
    assert client.post("/api/taxonomy/classes", json={"name": "x", "hier": "flat", "color": [1, 2, 3],
                                                      "parent": 999}).status_code == 422
    # custom class can be painted
    new = base_mask(False)
    new[300:310, 0:10] = 124
    assert client.put("/api/frames/A_000002/mask", content=png_gray(new), headers={"content-type": "image/png"}).status_code == 200
    pal = Image.open(env.master / "masks/train/A_000002.png").getpalette()
    assert pal[124 * 3:124 * 3 + 3] == [1, 2, 3]


def test_ids_exhausted(client):
    st = client.app.state.get_store()
    from common.taxonomy import ClassDef, Taxonomy
    extra = [ClassDef(i, f"c{i}", f"hackeye--x--c{i}", "x", (1, 1, 1)) for i in range(124, 255)]
    st.tax = Taxonomy(st.tax.name, st.tax.source, [*st.tax.classes, *extra])
    assert client.post("/api/taxonomy/classes", json={"name": "late", "hier": "x", "color": [1, 1, 1]}).status_code == 409


def test_split_patch_delete(client):
    r = client.post(f"/api/taxonomy/classes/{SIDEWALK}/split",
                    json={"children": [{"name": "sw-a", "color": [10, 0, 0]}, {"name": "sw-b", "color": [20, 0, 0]}]})
    assert r.status_code == 200
    ids = [c["id"] for c in r.json()["created"]]
    assert ids == [124, 125]
    c = by_id(client, 124)
    assert c["parent"] == SIDEWALK and c["hier"] == by_id(client, SIDEWALK)["hier"]
    assert client.post("/api/taxonomy/classes/999/split", json={"children": [{"name": "q", "color": [1, 1, 1]}]}).status_code == 404
    assert client.post(f"/api/taxonomy/classes/{SIDEWALK}/split", json={"children": []}).status_code == 422
    # patch
    p = client.patch("/api/taxonomy/classes/124", json={"name": "sidewalk A", "color": [9, 9, 9], "hier": "flat2"})
    assert p.status_code == 200 and by_id(client, 124)["name"] == "sidewalk A"
    assert client.patch("/api/taxonomy/classes/124", json={"name": "sw-b"}).status_code == 409
    assert client.patch(f"/api/taxonomy/classes/{ROAD}", json={"name": "x"}).status_code == 403
    assert client.patch("/api/taxonomy/classes/200", json={"name": "x"}).status_code == 404
    # delete guard
    assert client.delete(f"/api/taxonomy/classes/{ROAD}").status_code == 403
    assert client.delete("/api/taxonomy/classes/124").json() == {"deleted": 124}
    assert all(c["id"] != 124 for c in client.get("/api/taxonomy").json()["classes"])
    assert client.delete("/api/taxonomy/classes/124").status_code == 404


def test_delete_with_pixels_requires_remap(client, env):
    client.post("/api/taxonomy/classes", json={"name": "tmp", "hier": "x", "color": [5, 5, 5]})
    new = base_mask(False)
    new[300:330, 0:100] = 124
    client.put("/api/frames/A_000002/mask", content=png_gray(new), headers={"content-type": "image/png"})
    assert by_id(client, 124)["pixels"] == 3000
    r = client.delete("/api/taxonomy/classes/124")
    assert r.status_code == 409
    assert client.delete("/api/taxonomy/classes/124", params={"remap_to": 124}).status_code == 422
    assert client.delete("/api/taxonomy/classes/124", params={"remap_to": 200}).status_code == 422
    j = wait_job(client, client.delete("/api/taxonomy/classes/124", params={"remap_to": ROAD}).json()["job"])
    assert j["status"] == "done", j
    assert all(c["id"] != 124 for c in client.get("/api/taxonomy").json()["classes"])
    got = np.array(Image.open(env.master / "masks/train/A_000002.png"))
    assert not (got == 124).any() and (got[300:330, 0:100] == ROAD).all()


def test_maps_list_get_preview(client):
    assert client.get("/api/maps").json() == [{"name": "test-map", "n_classes": 124 - 4 + 2}]
    m = client.get("/api/maps/test-map").json()
    assert m["groups"]["wet"] == ["nature--snow", "nature--water"]
    assert client.get("/api/maps/nope").status_code == 404
    assert client.get("/api/maps/..%2Fx").status_code in (404, 422)
    p = client.post("/api/maps/preview", json={"groups": m["groups"]}).json()
    assert p["n"] == 122 and len(p["lut"]) == 256 and p["lut"][255] == 255
    assert p["lut"][ROAD] == p["lut"][SIDEWALK] and p["lut"][SNOW] == p["lut"][WATER]
    assert abs(sum(p["pixel_share"].values()) - 1.0) < 1e-9
    wet = next(c for c in p["classes"] if c["name"] == "wet")
    assert wet["members"] == [SNOW, WATER] and not wet["void"]
    assert ROAD not in p["unassigned"] and 0 in p["unassigned"]
    assert client.post("/api/maps/preview", json={"groups": {"x": ["nope--*"]}}).status_code == 422
    assert client.post("/api/maps/preview", json={"groups": {"x": ["nature--snow"], "y": ["nature--*"]}}).status_code == 422


def test_preview_ignore_group(client):
    p = client.post("/api/maps/preview", json={"groups": {"__ignore__": ["nature--snow"]}}).json()
    assert p["n"] == 123 and p["lut"][SNOW] == 255
    assert p["pixel_share"]["255"] > 0 and SNOW not in p["unassigned"]


def test_save_map(client, env):
    g = {"road-ish": ["construction--flat--road"]}
    assert client.post("/api/maps", json={"name": "my.map-1", "groups": g, "comment": "hello"}).json() == {"name": "my.map-1", "n_classes": 124}
    text = (env.repo / "taxonomies" / "my.map-1.yaml").read_text()
    assert text.startswith("# hello") and "road-ish" in text
    assert client.post("/api/maps", json={"name": "my.map-1", "groups": g}).status_code == 409
    assert client.post("/api/maps", json={"name": "my.map-1", "groups": {"r": ["construction--flat--*"]},
                                          "overwrite": True}).status_code == 200
    for bad in ("../x", "-x", "a b", ""):
        assert client.post("/api/maps", json={"name": bad, "groups": g}).status_code == 422
    assert client.post("/api/maps", json={"name": "bad", "groups": {"x": ["zzz"]}}).status_code == 422
    assert not (env.repo / "taxonomies" / "bad.yaml").exists()
    assert {m["name"] for m in client.get("/api/maps").json()} == {"test-map", "my.map-1"}
