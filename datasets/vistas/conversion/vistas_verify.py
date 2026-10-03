"""Correctness checks for the Vistas -> Ultralytics conversion, meant to run
BEFORE any GPU time is spent training. Each subcommand is one of the gates
from the plan doc's Verification section.

Usage:
    python vistas_verify.py roundtrip  --src RAW --dst CONVERTED --n 200
    python vistas_verify.py shapes     --dst CONVERTED
    python vistas_verify.py overlay    --src RAW --dst CONVERTED --n 12 --out OUT_DIR
    python vistas_verify.py yaml-check --yaml CONVERTED/vistas-v2.0.yaml
    python vistas_verify.py retention  --src RAW --short-sides 640,768,1024 --n 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from vistas_config import load_taxonomy
from vistas_convert import _SRC_SPLIT_DIR, _find_image, _target_size


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode not in ("P", "L"):
            raise ValueError(f"{path}: unexpected mode {im.mode!r}")
        return np.asarray(im)


def cmd_roundtrip(args: argparse.Namespace) -> None:
    """Passthrough-mode (--short-side 0) masks must be byte-identical to the
    source. This proves the palette/index handling end to end with zero
    resize-related ambiguity."""
    src, dst = Path(args.src), Path(args.dst)
    failures = []
    checked = 0
    for split in ("train", "val"):
        masks_dir = dst / "masks" / split
        if not masks_dir.is_dir():
            continue
        stems = [p.stem for p in masks_dir.glob("*.png")]
        random.shuffle(stems)
        for stem in stems[: args.n]:
            conv = _load_mask(masks_dir / f"{stem}.png")
            src_mask_path = src / _SRC_SPLIT_DIR[split] / "v2.0" / "labels" / f"{stem}.png"
            if not src_mask_path.exists():
                failures.append(f"{split}/{stem}: source mask missing at {src_mask_path}")
                continue
            source = _load_mask(src_mask_path)
            checked += 1
            if conv.shape == source.shape:
                if not np.array_equal(conv, source):
                    failures.append(f"{split}/{stem}: pixel mismatch (same shape)")
            else:
                # Not passthrough mode -- resize was applied. Converted classes
                # must be a subset of source classes; nearest-neighbour can
                # only drop rare classes at a boundary, never invent a new id.
                conv_ids, src_ids = set(np.unique(conv).tolist()), set(np.unique(source).tolist())
                extra = conv_ids - src_ids
                if extra:
                    failures.append(f"{split}/{stem}: converted has class ids {extra} absent from source")

    print(f"[verify:roundtrip] checked {checked} pairs, {len(failures)} failures", file=sys.stderr)
    for f in failures[:30]:
        print(f"  FAIL: {f}", file=sys.stderr)
    if failures:
        sys.exit(1)


def cmd_shapes(args: argparse.Namespace) -> None:
    """Shape lockstep + range check across the WHOLE converted dataset --
    exactly what Ultralytics' verify_image_mask will assert at scan time.
    Fail here in minutes, not 40 minutes into a training run."""
    dst = Path(args.dst)
    yaml_path = dst / "vistas-v2.0.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    max_id = max(int(k) for k in data["names"])

    failures = []
    total = 0
    for split in ("train", "val"):
        images_dir, masks_dir = dst / "images" / split, dst / "masks" / split
        if not images_dir.is_dir():
            continue
        for img_path in images_dir.glob("*.jpg"):
            total += 1
            mask_path = masks_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                failures.append(f"{split}/{img_path.stem}: no matching mask")
                continue
            with Image.open(img_path) as im:
                img_shape = (im.height, im.width)
            mask = _load_mask(mask_path)
            if img_shape != mask.shape:
                failures.append(f"{split}/{img_path.stem}: image {img_shape} != mask {mask.shape}")
            m = mask[mask != 255]
            if m.size and m.max() > max_id:
                failures.append(f"{split}/{img_path.stem}: mask id {m.max()} > max valid id {max_id}")

    print(f"[verify:shapes] checked {total} pairs, {len(failures)} failures", file=sys.stderr)
    for f in failures[:30]:
        print(f"  FAIL: {f}", file=sys.stderr)
    if failures:
        sys.exit(1)


def cmd_overlay(args: argparse.Namespace) -> None:
    """Blend N images with their palette-colorized masks for a human look --
    specifically at whether `curb` traces the sidewalk/road boundary."""
    src, dst, out = Path(args.src), Path(args.dst), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(src / "config_v2.0.json")

    sampled = 0
    for split in ("train", "val"):
        images_dir, masks_dir = dst / "images" / split, dst / "masks" / split
        if not images_dir.is_dir():
            continue
        stems = [p.stem for p in images_dir.glob("*.jpg")]
        random.shuffle(stems)
        for stem in stems[: args.n // 2 + 1]:
            img = cv2.imread(str(images_dir / f"{stem}.jpg"))
            with Image.open(masks_dir / f"{stem}.png") as m:
                mask = np.asarray(m.convert("P"))
            color = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for cid in np.unique(mask):
                if cid == 255:
                    continue
                color[mask == cid] = taxonomy.palette.get(int(cid), (0, 0, 0))[::-1]  # RGB -> BGR
            blend = cv2.addWeighted(img, 0.5, color, 0.5, 0)
            cv2.imwrite(str(out / f"{split}_{stem}_overlay.jpg"), blend)
            sampled += 1
            if sampled >= args.n:
                break
        if sampled >= args.n:
            break

    print(f"[verify:overlay] wrote {sampled} overlays to {out}", file=sys.stderr)
    print("[verify:overlay] look specifically at curb: it should trace a thin "
          "continuous ribbon exactly along the sidewalk/road boundary.", file=sys.stderr)


def cmd_yaml_check(args: argparse.Namespace) -> None:
    """The masks_dir-vs-PolygonSemanticDataset trap: build.py picks the mask
    dataset only if masks_dir is present in the YAML. Confirm it's there and
    that nc derives to what we expect, without needing a running trainer."""
    yaml_path = Path(args.yaml)
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    problems = []
    if "masks_dir" not in data or not data["masks_dir"]:
        problems.append("masks_dir missing/empty -- this YAML would select PolygonSemanticDataset, "
                         "which injects an extra 'background' class and silently bumps nc by one.")
    if "nc" in data:
        problems.append("nc is explicitly set -- leave it out, let it derive from len(names), "
                         "or a future taxonomy edit can silently desync from it.")
    for key in ("train", "val", "path", "names"):
        if key not in data:
            problems.append(f"missing required key: {key}")

    n_names = len(data.get("names", {}))
    print(f"[verify:yaml-check] {yaml_path}: masks_dir={data.get('masks_dir')!r}, "
          f"{n_names} names, path={data.get('path')!r}", file=sys.stderr)

    try:
        from ultralytics.data.utils import check_det_dataset
    except ImportError:
        print("[verify:yaml-check] ultralytics not importable here -- skipped check_det_dataset call, "
              "only did the static YAML checks above.", file=sys.stderr)
    else:
        try:
            checked = check_det_dataset(str(yaml_path))
            print(f"[verify:yaml-check] check_det_dataset: nc={checked.get('nc')}", file=sys.stderr)
            if checked.get("nc") != n_names:
                problems.append(f"check_det_dataset derived nc={checked.get('nc')} != len(names)={n_names}")
        except Exception as e:  # noqa: BLE001 -- a bad YAML raising here IS one of the problems we're reporting
            problems.append(f"check_det_dataset raised {type(e).__name__}: {e}")

    for p in problems:
        print(f"  PROBLEM: {p}", file=sys.stderr)
    if problems:
        sys.exit(1)


def cmd_retention(args: argparse.Namespace) -> None:
    """Per-class pixel-share retention at candidate --short-side values vs
    native resolution. Bulk classes should move <2%; thin classes (curb,
    curb cut, lane marking, pole) will lose share under nearest-neighbour
    downsampling -- pick --short-side at the point curb retention degrades
    sharply. This determines whether the trained model can see the thing
    this whole project cares about; trust this over any published table."""
    src = Path(args.src)
    taxonomy = load_taxonomy(src / "config_v2.0.json")
    short_sides = [int(s) for s in args.short_sides.split(",")]

    labels_dir = src / _SRC_SPLIT_DIR[args.split] / "v2.0" / "labels"
    stems = sorted(p.stem for p in labels_dir.glob("*.png"))
    random.seed(0)
    random.shuffle(stems)
    stems = stems[: args.n]

    counts = {0: np.zeros(taxonomy.n, dtype=np.int64)}
    for s in short_sides:
        counts[s] = np.zeros(taxonomy.n, dtype=np.int64)

    for stem in stems:
        mask = _load_mask(labels_dir / f"{stem}.png")
        h, w = mask.shape
        counts[0] += np.bincount(mask.ravel(), minlength=taxonomy.n)[: taxonomy.n]
        for s in short_sides:
            new_w, new_h = _target_size(w, h, s, args.max_long_side)
            resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            counts[s] += np.bincount(resized.ravel(), minlength=taxonomy.n)[: taxonomy.n]

    native_total = counts[0].sum()
    rows = []
    for cid in range(taxonomy.n):
        native_share = counts[0][cid] / native_total if native_total else 0.0
        row = {"id": cid, "name": taxonomy.names[cid], "native_share": native_share, "native_px": int(counts[0][cid])}
        for s in short_sides:
            total_s = counts[s].sum()
            share_s = counts[s][cid] / total_s if total_s else 0.0
            rel_change = (share_s - native_share) / native_share if native_share > 0 else float("nan")
            row[f"share_{s}"] = share_s
            row[f"rel_change_{s}"] = rel_change
        rows.append(row)

    rows.sort(key=lambda r: -r["native_share"])

    print(f"[verify:retention] {args.n} images sampled from {args.split}", file=sys.stderr)
    header = "class".ljust(28) + "native%".rjust(9) + "".join(f"  @{s}px_rel".rjust(12) for s in short_sides)
    print(header, file=sys.stderr)
    for r in rows:
        if r["native_share"] < args.min_share and not any(k in r["name"] for k in ("curb", "pole", "marking")):
            continue
        line = f"{r['name'][:27]:<28}{r['native_share']*100:8.3f}%"
        for s in short_sides:
            rc = r[f"rel_change_{s}"]
            line += f"  {rc*100:+10.1f}%" if rc == rc else "        n/a"
        print(line, file=sys.stderr)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[verify:retention] wrote {args.out}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("roundtrip")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--n", type=int, default=200)
    p.set_defaults(func=cmd_roundtrip)

    p = sub.add_parser("shapes")
    p.add_argument("--dst", required=True)
    p.set_defaults(func=cmd_shapes)

    p = sub.add_parser("overlay")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=12)
    p.set_defaults(func=cmd_overlay)

    p = sub.add_parser("yaml-check")
    p.add_argument("--yaml", required=True)
    p.set_defaults(func=cmd_yaml_check)

    p = sub.add_parser("retention")
    p.add_argument("--src", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--short-sides", default="640,768,1024")
    p.add_argument("--max-long-side", type=int, default=2048)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--min-share", type=float, default=0.001, help="hide classes below this native share unless curb/pole/marking")
    p.add_argument("--out", default=None, help="optional path to write full per-class JSON")
    p.set_defaults(func=cmd_retention)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
