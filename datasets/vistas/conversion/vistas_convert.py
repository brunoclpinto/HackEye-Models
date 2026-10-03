"""Converts Mapillary Vistas v2.0 into the on-disk layout Ultralytics' YOLO
semantic-segmentation trainer expects.

This is a file-layout + resize job, not a decode job: Vistas' v2.0/labels/*.png
are already 8-bit palette PNGs whose pixel values are indices 0..123 into
config_v2.0.json, which is exactly what Ultralytics' SemanticDataset wants
(verified against ultralytics 8.4.167 source, see plan doc). No pixel
remapping happens by default.

Output tree (this is what selects SemanticDataset over PolygonSemanticDataset
-- see the masks_dir key written into the YAML in vistas_config.py):

    <dst>/
      vistas-v2.0.yaml
      images/{train,val}/<stem>.jpg
      masks/{train,val}/<stem>.png
      conversion_manifest.json

The image path must contain a literal "/images/" component and the mask
stem+extension is derived from it (img2label_paths string-replaces the last
"/images/" with "/<masks_dir>/" and forces ".png") -- this tree satisfies that
by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from vistas_config import IGNORE_LABEL, VistasTaxonomy, load_taxonomy

# Vistas' on-disk split directory names differ from the train/val names
# Ultralytics' YAML convention (and our dst tree) uses.
_SRC_SPLIT_DIR = {"train": "training", "val": "validation"}
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class ConvertResult:
    stem: str
    ok: bool
    shape: tuple[int, int] | None
    msg: str


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in _IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _target_size(w: int, h: int, short_side: int, max_long_side: int) -> tuple[int, int]:
    """Returns (new_w, new_h). short_side <= 0 means passthrough (caller
    should symlink instead of calling this)."""
    short, long_ = (w, h) if w < h else (h, w)
    scale = short_side / short
    new_short = short_side
    new_long = round(long_ * scale)
    if new_long > max_long_side:
        scale *= max_long_side / new_long
        new_long = max_long_side
        new_short = round(short * scale)
    return (new_short, new_long) if w < h else (new_long, new_short)


def _build_palette_bytes(palette: dict[int, tuple[int, int, int]], n: int) -> list[int]:
    flat: list[int] = []
    for cid in range(n):
        flat.extend(palette[cid])
    flat.extend([0, 0, 0] * (256 - n))
    return flat[: 256 * 3]


def _convert_one(
    stem: str,
    src_img: Path,
    src_mask: Path,
    dst_img: Path,
    dst_mask: Path,
    short_side: int,
    max_long_side: int,
    palette: list[int],
    remap_lut: np.ndarray | None,
) -> ConvertResult:
    try:
        with Image.open(src_mask) as m:
            if m.mode not in ("P", "L"):
                return ConvertResult(stem, False, None, f"mask mode {m.mode!r}, expected P or L")
            mask = np.asarray(m)

        img_bgr = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return ConvertResult(stem, False, None, "cv2.imread failed (corrupt/unsupported image)")

        if remap_lut is not None:
            mask = remap_lut[mask]

        if short_side > 0:
            h, w = img_bgr.shape[:2]
            new_w, new_h = _target_size(w, h, short_side, max_long_side)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        if img_bgr.shape[:2] != mask.shape[:2]:
            return ConvertResult(
                stem, False, None,
                f"shape mismatch after resize: image {img_bgr.shape[:2]} vs mask {mask.shape[:2]}",
            )

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_mask.parent.mkdir(parents=True, exist_ok=True)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(dst_img, format="JPEG", quality=92)

        mask_im = Image.fromarray(mask.astype(np.uint8), mode="P")
        mask_im.putpalette(palette)
        mask_im.save(dst_mask, format="PNG")

        return ConvertResult(stem, True, img_bgr.shape[:2], "ok")
    except Exception as e:  # noqa: BLE001 -- one bad file must not kill an 18k-image batch
        return ConvertResult(stem, False, None, f"{type(e).__name__}: {e}")


def _symlink_one(stem: str, src_img: Path, src_mask: Path, dst_img: Path, dst_mask: Path) -> ConvertResult:
    try:
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_mask.parent.mkdir(parents=True, exist_ok=True)
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        if dst_mask.exists() or dst_mask.is_symlink():
            dst_mask.unlink()
        dst_img.symlink_to(src_img.resolve())
        dst_mask.symlink_to(src_mask.resolve())
        with Image.open(src_img) as im:
            shape = (im.height, im.width)
        return ConvertResult(stem, True, shape, "symlinked")
    except Exception as e:  # noqa: BLE001
        return ConvertResult(stem, False, None, f"{type(e).__name__}: {e}")


def _build_remap_lut(
    taxonomy: VistasTaxonomy, drop_void: bool
) -> tuple[np.ndarray | None, dict[int, str], dict[int, tuple[int, int, int]]]:
    """Returns (lut, new_names, new_palette). lut is None if no remapping is
    needed (the default: keep Vistas' native 0..123 indices for index-parity
    with the official taxonomy, and its palette is used as-is). With
    drop_void, renumbers to drop void classes entirely (nc goes from 124 to
    124 - len(void_ids), mapped to IGNORE_LABEL) -- the palette must be
    renumbered in lockstep, or saved masks would carry the OLD id's color at
    the NEW id's position and mislead anyone eyeballing them (Ultralytics
    itself only reads the raw index, never the palette, so this only affects
    human debugging, not training correctness -- but it's cheap to get right).
    """
    if not drop_void:
        return None, dict(taxonomy.names), dict(taxonomy.palette)

    lut = np.full(256, IGNORE_LABEL, dtype=np.uint8)
    new_names: dict[int, str] = {}
    new_palette: dict[int, tuple[int, int, int]] = {}
    new_id = 0
    for old_id in range(taxonomy.n):
        if old_id in taxonomy.void_ids:
            continue
        lut[old_id] = new_id
        new_names[new_id] = taxonomy.names[old_id]
        new_palette[new_id] = taxonomy.palette[old_id]
        new_id += 1
    return lut, new_names, new_palette


def convert_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    palette: dict[int, tuple[int, int, int]],
    n_classes: int,
    short_side: int,
    max_long_side: int,
    workers: int,
    limit: int | None,
    dry_run: bool,
    remap_lut: np.ndarray | None,
) -> tuple[int, int, list[str]]:
    src_split_dir = src_root / _SRC_SPLIT_DIR[split]
    images_dir = src_split_dir / "images"
    labels_dir = src_split_dir / "v2.0" / "labels"

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"expected labels at {labels_dir} -- re-check the extracted archive layout")

    stems = sorted(p.stem for p in labels_dir.glob("*.png"))
    if limit is not None:
        stems = stems[:limit]

    pairs: list[tuple[str, Path, Path]] = []
    missing_images: list[str] = []
    for stem in stems:
        img = _find_image(images_dir, stem)
        if img is None:
            missing_images.append(stem)
            continue
        pairs.append((stem, img, labels_dir / f"{stem}.png"))

    print(f"[vistas_convert] {split}: {len(stems)} masks, {len(pairs)} paired with an image, "
          f"{len(missing_images)} missing image", file=sys.stderr)

    if dry_run:
        return len(pairs), 0, missing_images

    palette_bytes = _build_palette_bytes(palette, n_classes)
    dst_images = dst_root / "images" / split
    dst_masks = dst_root / "masks" / split

    ok_count = 0
    errors: list[str] = list(f"missing image: {s}" for s in missing_images)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        if short_side > 0:
            futures = [
                pool.submit(
                    _convert_one, stem, img, mask, dst_images / f"{stem}.jpg", dst_masks / f"{stem}.png",
                    short_side, max_long_side, palette_bytes, remap_lut,
                )
                for stem, img, mask in pairs
            ]
        else:
            futures = [
                pool.submit(_symlink_one, stem, img, mask, dst_images / f"{stem}.jpg", dst_masks / f"{stem}.png")
                for stem, img, mask in pairs
            ]

        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r.ok:
                ok_count += 1
            else:
                errors.append(f"{r.stem}: {r.msg}")
            if i % 2000 == 0:
                print(f"[vistas_convert] {split}: {i}/{len(pairs)}", file=sys.stderr)

    print(f"[vistas_convert] {split}: {ok_count}/{len(pairs)} converted, {len(errors)} errors", file=sys.stderr)
    return ok_count, len(errors), errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="Vistas raw root (contains config_v2.0.json)")
    ap.add_argument("--dst", required=True, type=Path, help="output root for the converted dataset")
    ap.add_argument("--short-side", type=int, default=768, help="0 = passthrough (symlink, no resize)")
    ap.add_argument("--max-long-side", type=int, default=2048)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="cap images per split, for building a smoke set")
    ap.add_argument("--drop-void", action="store_true", help="renumber away void/ignore classes instead of keeping native 0..123 ids")
    ap.add_argument("--dry-run", action="store_true", help="count what would be converted, write nothing")
    args = ap.parse_args()

    config_path = args.src / "config_v2.0.json"
    if not config_path.is_file():
        sys.exit(f"[vistas_convert] FATAL: {config_path} not found -- is --src the Vistas raw root?")

    taxonomy = load_taxonomy(config_path)
    print(f"[vistas_convert] taxonomy: {taxonomy.n} classes, {len(taxonomy.void_ids)} void", file=sys.stderr)

    remap_lut, names, palette = _build_remap_lut(taxonomy, args.drop_void)

    splits = args.splits.split(",")
    args.dst.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": str(args.src.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "short_side": args.short_side,
        "max_long_side": args.max_long_side,
        "drop_void": args.drop_void,
        "n_classes": len(names),
        "void_ids": [] if args.drop_void else taxonomy.void_ids,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": {},
    }

    for split in splits:
        ok, err, errors = convert_split(
            args.src, args.dst, split, palette, len(names), args.short_side, args.max_long_side,
            args.workers, args.limit, args.dry_run, remap_lut,
        )
        manifest["splits"][split] = {"converted": ok, "errors": err, "error_detail": errors[:50]}

    if args.dry_run:
        print("[vistas_convert] dry run, nothing written", file=sys.stderr)
        return

    data_yaml_dict = {
        "path": str(args.dst.resolve()),
        "train": "images/train",
        "val": "images/val",
        "masks_dir": "masks",
        "names": names,
        "label_mapping": {} if args.drop_void else {vid: "ignore_label" for vid in taxonomy.void_ids},
    }
    yaml_path = args.dst / "vistas-v2.0.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml_dict, f, sort_keys=False)
    print(f"[vistas_convert] wrote {yaml_path}", file=sys.stderr)

    manifest_path = args.dst / "conversion_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[vistas_convert] wrote {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
