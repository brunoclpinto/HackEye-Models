import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from common import dataset_merge
from common.taxonomy import (
    ClassDef, Taxonomy, data_yaml_dict, from_vistas_config, load_merge_map, load_taxonomy_json,
    resolve_merge_map, save_merge_map, save_taxonomy_json,
)

VISTAS_CFG = Path("/datasets/vistas-raw/config_v2.0.json")
REPO = Path(__file__).resolve().parents[3]


def tiny_tax() -> Taxonomy:
    return Taxonomy("tiny", "test", [
        ClassDef(0, "road", "construction--flat--road", "construction", (128, 64, 128)),
        ClassDef(1, "sidewalk", "construction--flat--sidewalk", "construction", (244, 35, 232)),
        ClassDef(2, "car", "object--vehicle--car", "object", (0, 0, 142), void=True),
        ClassDef(3, "truck", "object--vehicle--truck", "object", (0, 0, 70)),
        ClassDef(4, "unlabeled", "void--unlabeled", "void", (0, 0, 0), void=True),
        ClassDef(124, "curb ramp", "hackeye--flat--curb-ramp", "hackeye", (1, 2, 3), parent=1),
    ])


def test_json_roundtrip_and_bak(tmp_path):
    tax = tiny_tax()
    p = tmp_path / "t.json"
    save_taxonomy_json(tax, p)
    assert not (tmp_path / "t.json.bak").exists()
    assert load_taxonomy_json(p) == tax
    first = p.read_text()
    tax.classes.append(ClassDef(125, "x", "hackeye--x", "hackeye", (9, 9, 9)))
    save_taxonomy_json(tax, p)
    assert (tmp_path / "t.json.bak").read_text() == first
    assert load_taxonomy_json(p).n == 7
    assert not list(tmp_path.glob("*.tmp"))


def test_rejects_bad_ids_and_dupes():
    with pytest.raises(ValueError):
        Taxonomy("a", "", [ClassDef(255, "i", "x--i", "x", (0, 0, 0))])
    with pytest.raises(ValueError):
        Taxonomy("a", "", [ClassDef(0, "a", "x--a", "x", (0, 0, 0)), ClassDef(1, "a", "x--b", "x", (0, 0, 0))])


def test_custom_ids_in_merge_map():
    tax = tiny_tax()
    m = resolve_merge_map(tax, {"flat": ["construction--flat--sidewalk", "hackeye--flat--*"],
                                "vehicle": ["object--vehicle--*"]})
    # dense ids by lowest original id: road 0, flat 1, vehicle 2, unlabeled 3
    assert m.names == {0: "road", 1: "flat", 2: "vehicle", 3: "unlabeled"}
    assert m.lut[124] == 1 and m.lut[1] == 1 and m.lut[2] == 2 and m.lut[3] == 2
    assert m.lut[255] == 255 and m.lut[100] == 255
    assert m.members[1] == [1, 124]
    assert m.void_ids == [3]            # car void is un-voided into vehicle; unlisted void stays void
    assert m.palette[1] == (244, 35, 232)   # first listed member's colour


def test_merge_errors():
    tax = tiny_tax()
    with pytest.raises(ValueError, match="matches no class"):
        resolve_merge_map(tax, {"g": ["nope--*"]})
    with pytest.raises(ValueError, match="claimed by both"):
        resolve_merge_map(tax, {"a": ["object--vehicle--car"], "b": ["object--vehicle--*"]})


def test_merge_map_roundtrip(tmp_path):
    groups = {"curb": ["construction--barrier--curb", "construction--flat--curb-cut"],
              "vehicle": ["object--vehicle--*"]}
    p = tmp_path / "m.yaml"
    save_merge_map(groups, p, "header line\nsecond")
    text = p.read_text()
    assert text.startswith("# header line\n# second\n")
    assert "\n  - object--vehicle--*\n" in text
    assert load_merge_map(p) == groups


def test_data_yaml_dict():
    d = data_yaml_dict({0: "a", 1: "b"}, [1], "/x")
    assert list(d) == ["path", "train", "val", "masks_dir", "names", "label_mapping"]
    assert d["label_mapping"] == {1: "ignore_label"} and "nc" not in d


def test_dataset_merge_roundtrip(tmp_path):
    tax = tiny_tax()
    src = tmp_path / "src"
    for sp in ("train", "val"):
        (src / "images" / sp).mkdir(parents=True)
        (src / "masks" / sp).mkdir(parents=True)
    rng = np.random.default_rng(0)
    ids = np.array([0, 1, 2, 3, 4, 124, 255], np.uint8)
    for sp in ("train", "val"):
        for i in range(3):
            Image.new("RGB", (16, 8)).save(src / "images" / sp / f"f{i}.jpg")
            m = Image.fromarray(rng.choice(ids, (8, 16)).astype(np.uint8), mode="P")
            m.putpalette(tax.palette_bytes())
            m.save(src / "masks" / sp / f"f{i}.png")
    (src / "src.yaml").write_text(yaml.safe_dump(data_yaml_dict(tax.names, tax.void_ids, str(src))))
    mp = tmp_path / "map.yaml"
    save_merge_map({"flat": ["construction--flat--sidewalk", "hackeye--flat--*"]}, mp)
    dst = tmp_path / "dst"
    ok = dataset_merge.run(src, dst, mp, tax, "sha", ["train", "val"], 2, False)
    assert ok
    out = yaml.safe_load((dst / "dst.yaml").read_text())
    assert out["names"] == {0: "road", 1: "flat", 2: "car", 3: "truck", 4: "unlabeled"}
    assert out["label_mapping"] == {2: "ignore_label", 4: "ignore_label"}
    man = json.loads((dst / "merge_manifest.json").read_text())
    assert man["splits"]["train"]["accounting_mismatches"] == []
    assert (src / "images/train/f0.jpg").stat().st_ino == (dst / "images/train/f0.jpg").stat().st_ino
    with Image.open(dst / "masks/train/f0.png") as m:
        assert m.mode == "P" and set(np.unique(np.asarray(m))) <= {0, 1, 2, 3, 4, 255}


@pytest.mark.skipif(not VISTAS_CFG.exists(), reason="Vistas config not available")
def test_lut_equals_build_merge_lut_on_vistas():
    import vistas_config as vc

    groups = load_merge_map(REPO / "taxonomies" / "HackEye26.1-48.yaml")
    old = vc.build_merge_lut(vc.load_taxonomy(VISTAS_CFG), groups)
    tax = from_vistas_config(VISTAS_CFG)
    new = resolve_merge_map(tax, groups)
    assert np.array_equal(old.lut, new.lut)
    assert (old.names, old.palette, old.void_ids, old.members) == (new.names, new.palette, new.void_ids, new.members)
    assert hashlib.sha256(new.lut.tobytes()).hexdigest() == "a84b7dd1f00ea42b1ade293d260ce6d5ce4ff30c4257e009e30c1671e1d58a78"
    # the shipped vistas124.json equals what the config generates
    assert load_taxonomy_json(REPO / "taxonomies" / "vistas124.json") == tax


def test_ignore_group_maps_to_255():
    tax = tiny_tax()
    m = resolve_merge_map(tax, {"flat": ["construction--flat--*"], "__ignore__": ["object--vehicle--truck"]})
    # flat (road+sidewalk), car, unlabeled, curb ramp; truck gets no output class
    assert m.n == 4
    assert m.ignored == [3]
    assert m.lut[3] == 255
    assert all(3 not in mem for mem in m.members.values())
    assert "__ignore__" not in m.names.values()
    assert sorted(m.names) == list(range(m.n))


def test_ignore_group_conflict_and_empty_match():
    tax = tiny_tax()
    with pytest.raises(ValueError, match="claimed by both"):
        resolve_merge_map(tax, {"a": ["object--vehicle--truck"], "__ignore__": ["object--vehicle--*"]})
    with pytest.raises(ValueError, match="matches no class"):
        resolve_merge_map(tax, {"__ignore__": ["nope--*"]})


def test_ignore_group_export_accounting(tmp_path):
    tax = tiny_tax()
    src = tmp_path / "src"
    for d in ("images", "masks"):
        for sp in ("train", "val"):
            (src / d / sp).mkdir(parents=True)
    arr = np.zeros((8, 8), np.uint8)
    arr[:, 4:] = 3
    arr[0, 0] = 255
    im = Image.fromarray(arr, "P")
    im.putpalette(tax.palette_bytes())
    im.save(src / "masks" / "train" / "a.png")
    Image.new("RGB", (8, 8)).save(src / "images" / "train" / "a.jpg")
    (src / "src.yaml").write_text(yaml.safe_dump(data_yaml_dict(tax.names, tax.void_ids, str(src))))
    mp = tmp_path / "m.yaml"
    save_merge_map({"__ignore__": ["object--vehicle--truck"]}, mp)
    assert dataset_merge.run(src, tmp_path / "dst", mp, tax, "x", ["train"], 1, False)
    out = np.asarray(Image.open(tmp_path / "dst" / "masks" / "train" / "a.png"))
    assert (out == 255).sum() == 33


def test_vistas124_names_unique_and_readable():
    """v26.1.7: every class name is unique and none is a raw Vistas id ('--')."""
    names = [c.name for c in load_taxonomy_json(REPO / "taxonomies" / "vistas124.json").classes]
    assert len(names) == len(set(names))
    assert not [n for n in names if "--" in n]


def test_unique_names_collision_rule():
    import sys
    sys.path.insert(0, str(REPO / "datasets" / "vistas" / "conversion"))
    from vistas_config import _short_name, _unique_names
    raw = {0: "marking--continuous--dashed", 1: "marking-only--continuous--dashed", 2: "construction--flat--road", 3: "a--b--c", 4: "x--b--c"}
    out = _unique_names(raw, {i: _short_name(r) for i, r in raw.items()})
    assert out[0] == "marking dashed" and out[1] == "marking-only dashed" and out[2] == "flat road"
    assert out[3] == "a c" and out[4] == "x c"      # distinct first parts: first + last
    raw = {0: "a--b--c", 1: "a--x--b--c"}           # short and first + last both collide: all parts
    assert _unique_names(raw, {i: _short_name(r) for i, r in raw.items()}) == {0: "a b c", 1: "a x b c"}
