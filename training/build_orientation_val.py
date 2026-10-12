"""Build orientation-test variants of a merged val split at production scale.

Simulates portrait inference from landscape Vistas frames (Vistas has almost
no portrait images: 21/18000 train, 5/2000 val). Every variant scales the
short side to --short (540 = half 1080p), then:

  landscape  full frame (e.g. 960x540) -- baseline
  center     center crop, width = --short (540x540), sides dropped
  portrait   center crop + sky-colored padding on top to --tall (540x960)

The padded rows are filled with the image's mean GT-sky color (fallback: a
neutral sky blue) and get mask value 255, which diagnose_surfaces.py ignores,
so scores cover only real pixels. Each variant gets its own dataset yaml for
diagnose_surfaces.py --match-train-scale --imgsz <short>.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

FALLBACK_SKY_BGR = (205, 170, 135)
VARIANTS = ("landscape", "center", "portrait")


def _one(job):
    img_p, mask_p, dst, short, tall, sky_id = job
    img = cv2.imread(str(img_p))
    mask = np.asarray(Image.open(mask_p))
    h, w = mask.shape
    k = short / min(h, w)
    size = (round(w * k), round(h * k))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    sky = mask == sky_id
    color = img[sky].mean(0).round().astype(np.uint8) if sky.any() else np.array(FALLBACK_SKY_BGR, np.uint8)

    x0 = (img.shape[1] - short) // 2
    crop_i, crop_m = img[-tall:, x0 : x0 + short], mask[-tall:, x0 : x0 + short]  # [-tall:] only bites on rare tall portrait sources
    pad = tall - crop_i.shape[0]
    out = {
        "landscape": (img, mask),
        "center": (crop_i, crop_m),
        "portrait": (
            np.vstack([np.broadcast_to(color, (pad, short, 3)), crop_i]),
            np.vstack([np.full((pad, short), 255, np.uint8), crop_m]),
        ),
    }
    for v, (i, m) in out.items():
        cv2.imwrite(str(dst / v / "images/val" / (img_p.stem + ".jpg")), i, [cv2.IMWRITE_JPEG_QUALITY, 95])
        Image.fromarray(np.ascontiguousarray(m)).save(dst / v / "masks/val" / (img_p.stem + ".png"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="merged dataset root (has the yaml, images/val, masks/val)")
    ap.add_argument("--dst", required=True, type=Path, help="output root; one subfolder per variant")
    ap.add_argument("--yaml-name", default="vistas-v2.0.yaml")
    ap.add_argument("--short", type=int, default=540)
    ap.add_argument("--tall", type=int, default=960)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    data = yaml.safe_load((args.src / args.yaml_name).read_text())
    sky_id = next(int(k) for k, v in data["names"].items() if v == "nature sky")
    for v in VARIANTS:
        (args.dst / v / "images/val").mkdir(parents=True, exist_ok=True)
        (args.dst / v / "masks/val").mkdir(parents=True, exist_ok=True)
        (args.dst / v / args.yaml_name).write_text(yaml.safe_dump({**data, "path": str(args.dst / v)}, sort_keys=False))

    imgs = sorted(p for p in (args.src / "images/val").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    jobs = [(p, args.src / "masks/val" / (p.stem + ".png"), args.dst, args.short, args.tall, sky_id) for p in imgs]
    with ProcessPoolExecutor(args.workers) as ex:
        for n, _ in enumerate(ex.map(_one, jobs, chunksize=16), 1):
            if n % 500 == 0 or n == len(jobs):
                print(f"[orientation-val] {n}/{len(jobs)}", flush=True)
    print(f"[orientation-val] wrote {', '.join(VARIANTS)} under {args.dst}")


if __name__ == "__main__":
    main()
