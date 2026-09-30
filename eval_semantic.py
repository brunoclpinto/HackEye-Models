"""Reports the numbers this project actually cares about from a trained
checkpoint: not just headline mIoU, but per-class IoU for road/sidewalk/curb/
curb-cut specifically, plus what they're being confused with.

model.val() returns the SemanticMetrics instance directly (verified:
engine/model.py Model.val() does `self.metrics = validator.metrics; return
validator.metrics`), which already computes everything needed -- per-class
IoU, confusion matrix, DataExportMixin.to_csv() -- via polars, already an
ultralytics transitive dependency. No custom metric code here, just a report
wrapper around SemanticMetrics.
"""

from __future__ import annotations

import argparse
import sys

import yaml


def _matching_indices(names: dict[int, str], keywords: list[str]) -> list[tuple[int, str]]:
    """Substring match against the converter's class names. Note the
    converter's default naming (vistas_config._short_name) joins the last
    two '--'-hierarchy components, e.g. Vistas' 'construction--flat--curb'
    becomes 'flat curb' and 'construction--flat--curb-cut' becomes
    'flat curb-cut' -- NOT the bare 'curb'/'curb cut' this project talks
    about informally. Substring matching on 'curb' catches both, which
    is what we want; it is not an exact-name lookup."""
    out = []
    for idx, name in names.items():
        low = name.lower()
        if any(kw.lower() in low for kw in keywords):
            out.append((idx, name))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="val")
    ap.add_argument("--target-keywords", default="road,sidewalk,curb",
                     help="comma-separated substrings matched against class names (case-insensitive)")
    ap.add_argument("--out-csv", default=None, help="optional path to write the full 124-class summary as CSV")
    ap.add_argument("--project", default="/runs",
                     help="without this, model.val() defaults to a path under the container's own "
                          "filesystem (e.g. /training/runs/...), which is NOT the persistent runs "
                          "mount and vanishes when the container exits")
    ap.add_argument("--name", default="eval")
    args = ap.parse_args()

    from ultralytics import YOLO

    with open(args.data) as f:
        data = yaml.safe_load(f)
    names = {int(k): v for k, v in data["names"].items()}

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data, split=args.split, imgsz=args.imgsz, plots=True, save_json=True,
        project=args.project, name=args.name,
    )

    print(f"\n[eval_semantic] headline mIoU={metrics.miou:.4f}  pixel_acc={metrics.pixel_accuracy:.4f}  "
          f"(averaged only over the {len(metrics.ap_class_index)}/{metrics.nc} classes present in {args.split} GT)",
          file=sys.stderr)
    print("[eval_semantic] calibration: published Vistas v2.0 mIoU for heavyweight models on the full "
          "taxonomy is low-to-mid 40s. A ~10M-param model at this imgsz landing high-20s to mid-30s is "
          "reasonable -- judge this run on the target-class table below, not the headline number.",
          file=sys.stderr)

    targets = _matching_indices(names, args.target_keywords.split(","))
    if not targets:
        print(f"[eval_semantic] WARNING: no class names matched keywords {args.target_keywords!r}", file=sys.stderr)
    else:
        print(f"\n[eval_semantic] target classes (this is the real acceptance criterion, not mIoU):", file=sys.stderr)
        header = f"{'class':<30}{'IoU':>8}{'pixel_acc':>11}{'GT pixels':>14}{'images':>9}"
        print(header, file=sys.stderr)
        for idx, name in sorted(targets, key=lambda t: -metrics.per_class_iou[t[0]]):
            iou = metrics.per_class_iou[idx]
            pa = metrics.per_class_pixel_accuracy[idx]
            npix = int(metrics.nt_per_class[idx]) if idx < len(metrics.nt_per_class) else 0
            nimg = int(metrics.nt_per_image[idx]) if idx < len(metrics.nt_per_image) else 0
            flag = "  <-- near-zero, treat as failure for this project" if npix > 0 and iou < 0.10 else ""
            print(f"{name:<30}{iou:8.4f}{pa:11.4f}{npix:14,}{nimg:9,}{flag}", file=sys.stderr)

        if metrics.matrix is not None and len(targets) > 1:
            idxs = [i for i, _ in targets]
            sub = metrics.matrix[idxs][:, idxs].cpu().numpy()
            print(f"\n[eval_semantic] confusion sub-block (rows=ground truth, cols=predicted; "
                  f"row-normalized %):", file=sys.stderr)
            tgt_names = [n for _, n in targets]
            print(" " * 20 + "".join(f"{n[:14]:>15}" for n in tgt_names), file=sys.stderr)
            for r, name in enumerate(tgt_names):
                row_sum = sub[r].sum()
                row_pct = sub[r] / row_sum * 100 if row_sum > 0 else sub[r]
                print(f"{name[:19]:<20}" + "".join(f"{v:14.1f}%" for v in row_pct), file=sys.stderr)
            print("[eval_semantic] if curb -> sidewalk/road dominates, the model sees the boundary region "
                  "but can't localize the transition -- that argues for higher --imgsz, not more epochs.",
                  file=sys.stderr)

    if args.out_csv:
        with open(args.out_csv, "w") as f:
            f.write(metrics.to_csv())
        print(f"\n[eval_semantic] wrote full {metrics.nc}-class summary to {args.out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
