"""Diagnoses *why* the navigable-surface classes (road / sidewalk / bike-lane /
crosswalk / pedestrian-area) get confused, to decide where the next round of
GPU time goes. Two failure modes look identical in mIoU but need opposite fixes:

  1. Boundary failure -- errors concentrated within a few pixels of a GT
     boundary between surfaces. The model can't localize the thin cue (curb,
     paint line, wall) that separates them -> higher resolution / boundary
     crops.
  2. Region failure -- whole GT regions labeled as the wrong surface, far from
     any boundary. The model never propagated the cue across the surface ->
     context / capacity / class imbalance (oversampling, class weights,
     bigger model), not resolution.

Runs the model itself (model.predict) rather than model.val(), since val only
exposes the aggregated confusion matrix and this needs per-pixel errors with
their spatial position. Predictions come back at the stored mask resolution
(SemanticSegmentationPredictor rescales to orig_img), so all distances below
are in pixels at the converted dataset's resolution (short side 768 by
default), not at the 640 the network sees.

Writes a JSON report with everything and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

# Substrings matched against class names (same convention as eval_semantic.py).
SURFACE_KEYWORDS = [
    "flat road", "flat sidewalk", "flat bike-lane", "flat crosswalk-plain",
    "crosswalk-zebra", "flat pedestrian-area",
]
# Boundary cues: the things that separate same-looking pavement.
CUE_KEYWORDS = ["barrier curb", "flat curb-cut", "barrier wall", "barrier fence",
                "continuous solid", "continuous dashed", "marking--continuous--dashed"]

DIST_BUCKETS = [0, 3, 6, 12, 24, 48, 96, 10**9]  # px from nearest surface/cue boundary
REGION_MIN_PX = 400  # ignore tiny GT components; they're boundary slivers by definition


def _match(names: dict[int, str], keywords: list[str]) -> list[int]:
    out = []
    for kw in keywords:
        for idx, name in names.items():
            if kw.lower() in name.lower() and idx not in out:
                out.append(idx)
    return out


def _bucket_labels() -> list[str]:
    return [f"{a}-{b}px" if b < 10**9 else f"{a}+px" for a, b in zip(DIST_BUCKETS[:-1], DIST_BUCKETS[1:])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images in the split")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    from ultralytics import YOLO

    with open(args.data) as f:
        data = yaml.safe_load(f)
    names = {int(k): v for k, v in data["names"].items()}
    nc = len(names)
    void = {int(k) for k, v in (data.get("label_mapping") or {}).items() if v == "ignore_label"}
    root = Path(data["path"])
    img_dir = root / data[args.split]
    mask_dir = root / data.get("masks_dir", "masks") / Path(data[args.split]).name

    surf = _match(names, SURFACE_KEYWORDS)
    cues = _match(names, CUE_KEYWORDS)
    focus = surf + cues
    print(f"[diag] surfaces: {[names[i] for i in surf]}", file=sys.stderr)
    print(f"[diag] cues:     {[names[i] for i in cues]}", file=sys.stderr)

    # Remap: focus classes keep their own slot, everything else -> "other", void -> ignore.
    OTHER, IGN = len(focus), 255
    lut = np.full(256, OTHER, np.uint8)
    for j, c in enumerate(focus):
        lut[c] = j
    for v in void:
        lut[v] = IGN
    lut[255] = IGN
    labels = [names[c] for c in focus] + ["other"]
    K = len(labels)
    surf_slots = list(range(len(surf)))
    cue_slots = list(range(len(surf), len(focus)))
    # Boundaries that matter: any change of label among surfaces+cues, plus surface/other edges.
    boundary_slots = set(surf_slots + cue_slots)

    nb = len(DIST_BUCKETS) - 1
    conf = np.zeros((K, K), np.int64)
    # per GT surface slot, per distance bucket: [total, wrong, wrong-as-other-surface]
    dist_tot = np.zeros((len(surf), nb), np.int64)
    dist_err = np.zeros((len(surf), nb), np.int64)
    dist_err_surf = np.zeros((len(surf), nb), np.int64)
    # region outcomes per GT surface slot: counts + pixel mass
    reg_kinds = ["correct(>=70%)", "partial(30-70%)", "wrong(<30%)"]
    reg_n = np.zeros((len(surf), 3), np.int64)
    reg_px = np.zeros((len(surf), 3), np.int64)
    # For wholly-wrong regions: what surface was predicted instead (majority)
    reg_wrong_as = np.zeros((len(surf), K), np.int64)
    # Cue detection: for each GT cue slot, recall split near vs far from camera? keep simple: recall
    worst_images: list[tuple[float, str]] = []

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        images = images[: args.limit]
    model = YOLO(args.weights)

    for start in range(0, len(images), args.batch):
        chunk = images[start : start + args.batch]
        results = model.predict([str(p) for p in chunk], imgsz=args.imgsz, verbose=False, batch=args.batch)
        for p, r in zip(chunk, results):
            # Palette ("P") PNGs: cv2 would expand them to RGB colors; PIL gives the raw class ids.
            gt_raw = np.asarray(Image.open(mask_dir / (p.stem + ".png")))
            pr_raw = r.semantic_mask.data.cpu().numpy().astype(np.uint8)
            assert gt_raw.shape == pr_raw.shape, (p.name, gt_raw.shape, pr_raw.shape)
            gt, pr = lut[gt_raw], lut[pr_raw]
            valid = gt != IGN

            conf += np.bincount(gt[valid].astype(np.int64) * K + pr[valid], minlength=K * K).reshape(K, K)

            # Distance to nearest GT boundary (label change where at least one side is a surface/cue).
            g = gt.astype(np.int16)
            edge = np.zeros(gt.shape, bool)
            dh = g[:, 1:] != g[:, :-1]
            dv = g[1:, :] != g[:-1, :]
            edge[:, 1:] |= dh
            edge[:, :-1] |= dh
            edge[1:, :] |= dv
            edge[:-1, :] |= dv
            relevant = np.isin(gt, list(boundary_slots))
            edge &= cv2.dilate(relevant.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            dist = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
            bidx = np.digitize(dist, DIST_BUCKETS[1:-1])

            img_err, img_tot = 0, 0
            for s in surf_slots:
                m = (gt == s) & valid
                if not m.any():
                    continue
                wrong = m & (pr != s)
                wrong_s = wrong & np.isin(pr, surf_slots)
                dist_tot[s] += np.bincount(bidx[m], minlength=nb)
                dist_err[s] += np.bincount(bidx[wrong], minlength=nb)
                dist_err_surf[s] += np.bincount(bidx[wrong_s], minlength=nb)
                img_err += int(wrong_s.sum())
                img_tot += int(m.sum())

                n, cc = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
                if n <= 1:
                    continue
                area = np.bincount(cc.ravel(), minlength=n)
                hit = np.bincount(cc[pr == s].ravel(), minlength=n)
                for k in range(1, n):
                    if area[k] < REGION_MIN_PX:
                        continue
                    frac = hit[k] / area[k]
                    kind = 0 if frac >= 0.7 else 1 if frac >= 0.3 else 2
                    reg_n[s, kind] += 1
                    reg_px[s, kind] += area[k]
                    if kind == 2:
                        reg_wrong_as[s, np.bincount(pr[cc == k], minlength=K).argmax()] += 1
            if img_tot:
                worst_images.append((img_err / img_tot, p.stem))
        print(f"[diag] {min(start + args.batch, len(images))}/{len(images)}", file=sys.stderr, end="\r")
    print(file=sys.stderr)

    # ---------------- report ----------------
    bl = _bucket_labels()
    rown = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    iou = np.diag(conf) / np.maximum(conf.sum(0) + conf.sum(1) - np.diag(conf), 1)
    report = {
        "weights": args.weights, "split": args.split, "n_images": len(images), "imgsz": args.imgsz,
        "labels": labels, "surface_labels": [labels[s] for s in surf_slots],
        "confusion_counts": conf.tolist(), "confusion_row_norm": rown.round(4).tolist(),
        "iou_in_remapped_space": dict(zip(labels, iou.round(4).tolist())),
        "dist_buckets": bl,
        "dist_total": dist_tot.tolist(), "dist_wrong": dist_err.tolist(), "dist_wrong_as_surface": dist_err_surf.tolist(),
        "region_kinds": reg_kinds, "region_count": reg_n.tolist(), "region_pixels": reg_px.tolist(),
        "region_wrong_predicted_as": reg_wrong_as.tolist(),
        "worst_images_surface_confusion": [s for _, s in sorted(worst_images, reverse=True)[:40]],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=1))

    def short(n: str) -> str:
        return n.replace("flat ", "").replace("marking--discrete--", "").replace("barrier ", "")[:11]

    print("\n== Confusion (rows=GT, cols=pred, row %) ==")
    print(f"{'':<13}" + "".join(f"{short(l):>12}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{short(l):<13}" + "".join(f"{v * 100:11.1f}%" for v in rown[i]))

    print("\n== Surface error rate by distance to nearest GT boundary (all-wrong / wrong-as-another-surface) ==")
    print(f"{'':<13}" + "".join(f"{b:>15}" for b in bl))
    for s in surf_slots:
        cells = []
        for b in range(nb):
            t = dist_tot[s, b]
            cells.append(f"{dist_err[s, b] / t * 100:5.1f}/{dist_err_surf[s, b] / t * 100:4.1f}%" if t else "-")
        print(f"{short(labels[s]):<13}" + "".join(f"{c:>15}" for c in cells))
    print("\n== Where surface-vs-surface error pixels live (share of that class's surface-confusion errors) ==")
    print(f"{'':<13}" + "".join(f"{b:>10}" for b in bl))
    for s in surf_slots:
        tot = dist_err_surf[s].sum()
        print(f"{short(labels[s]):<13}" + "".join(f"{v / max(tot, 1) * 100:9.1f}%" for v in dist_err_surf[s]))

    print(f"\n== GT regions (>= {REGION_MIN_PX}px) by fraction predicted correctly: count (pixel share) ==")
    for s in surf_slots:
        tot_px = max(reg_px[s].sum(), 1)
        cells = "   ".join(f"{k}: {reg_n[s, i]} ({reg_px[s, i] / tot_px * 100:.1f}%)" for i, k in enumerate(reg_kinds))
        wrong_as = ", ".join(f"{short(labels[j])}={reg_wrong_as[s, j]}" for j in np.argsort(-reg_wrong_as[s])[:3]
                             if reg_wrong_as[s, j])
        print(f"{short(labels[s]):<13}{cells}" + (f"   | wrong regions became: {wrong_as}" if wrong_as else ""))

    print(f"\n[diag] full report -> {args.out_json}")


if __name__ == "__main__":
    main()
