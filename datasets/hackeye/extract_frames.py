"""Extracts the HackEye media library (GoPro MP4 clips) into 1280x720 JPG frames
at a fixed rate, laid out like the Vistas conversion so the pseudo-labeller
(pseudo_label.py) can add masks/ next to images/ without further reshuffling.

    <dst>/
      images/{train,val}/<clip>_<k:06d>.jpg
      meta/frames_<clip>.csv        one per clip: stem,split,clip,k,t_sec,src_frame,src_fps,src_w,src_h

Why it is built the way it is (see datasets/hackeye/PLAN.md, "Fixed decisions"):

* Frame pick is index-based, not time-seek based. For output index k the source
  frame is round(k / fps_out * src_fps). Seeking in h264 is both slow and
  inexact; instead the decoder walks the clip once, grab()bing (decode without
  the colour conversion/copy) every frame and retrieve()ing only the chosen
  ones. 59.94 fps sources thus keep every 3rd frame, 23.976 fps sources drop
  about 1 in 6, unevenly -- exactly what the plan specifies.
* The stem <clip>_<k:06d> makes t = k / fps_out recoverable from the name.
* Resize is INTER_AREA to exactly WxH (all sources are 16:9; 2704x1520 is a
  0.05% stretch, accepted in the plan).
* JPEG quality 100 + optimized Huffman is the maximum OpenCV offers; the plan
  budgets ~113 GB for it. Frames are the ground truth the curation works on,
  so they are not degraded further here.
* Resumable and crash-safe: every jpg and every csv is written to a temp name
  and os.replace()d, so a killed run never leaves a half-written file under a
  final name. A clip whose csv exists and whose jpg count matches the csv is
  skipped outright; otherwise existing jpgs are not re-encoded (the decoder
  still has to advance through them, but retrieve/resize/encode are skipped).
* One process per clip: decode is the bottleneck and clips are independent.

Run in the train container. docker/compose.yaml mounts the video library
(HACKEYE_MEDIA_DIR, default ~/media/hackeye) read-only at /media-src:

    cd docker && docker compose run --rm -T --entrypoint python train \\
        datasets/hackeye/extract_frames.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

CSV_COLS = ["stem", "split", "clip", "k", "t_sec", "src_frame", "src_fps", "src_w", "src_h"]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_csv_atomic(path: Path, rows: list[list]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        w.writerows(rows)
    os.replace(tmp, path)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _clip_done(csv_path: Path, img_dir: Path, clip: str) -> bool:
    if not csv_path.exists():
        return False
    with open(csv_path, newline="") as f:
        n_rows = sum(1 for _ in csv.DictReader(f))
    n_jpg = sum(1 for _ in img_dir.glob(f"{clip}_*.jpg"))
    return n_rows > 0 and n_rows == n_jpg


def extract_clip(src: str, dst: str, split: str, fps_out: float, width: int, height: int,
                 quality: int, limit_seconds: float | None) -> tuple[str, str, int, int, float]:
    """Returns (clip, split, frames, newly_encoded, elapsed_s)."""
    t0 = time.time()
    src_p, dst_p = Path(src), Path(dst)
    clip = src_p.stem
    img_dir = dst_p / "images" / split
    meta_dir = dst_p / "meta"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    csv_path = meta_dir / f"frames_{clip}.csv"

    if _clip_done(csv_path, img_dir, clip):
        with open(csv_path, newline="") as f:
            n = sum(1 for _ in csv.DictReader(f))
        _log(f"[extract] {clip} ({split}): already complete, {n} frames, skipped")
        return clip, split, n, 0, 0.0

    cap = cv2.VideoCapture(str(src_p))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src_p}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if src_fps <= fps_out:
        raise RuntimeError(f"{clip}: source fps {src_fps} <= output fps {fps_out}; this script only decimates")
    _log(f"[extract] {clip} ({split}): {src_w}x{src_h} @ {src_fps:.3f} fps, {n_src} source frames")

    params = [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    rows: list[list] = []
    pos = 0            # index of the next frame grab() will decode
    encoded = 0
    k = 0
    last_log = time.time()
    while True:
        t = k / fps_out
        if limit_seconds is not None and t >= limit_seconds:
            break
        target = round(t * src_fps)
        if target >= n_src:
            break
        ok = True
        while pos <= target:  # sequential decode; no seeking
            if not cap.grab():
                ok = False
                break
            pos += 1
        if not ok:
            _log(f"[extract] {clip}: decode failed at source frame {pos} (expected {n_src}); stopping at k={k}")
            break
        stem = f"{clip}_{k:06d}"
        final = img_dir / f"{stem}.jpg"
        if not final.exists():
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                _log(f"[extract] {clip}: retrieve failed at source frame {target}; stopping at k={k}")
                break
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small, params)
            if not ok:
                raise RuntimeError(f"{stem}: imencode failed")
            _atomic_write_bytes(final, buf.tobytes())
            encoded += 1
        rows.append([stem, split, clip, k, f"{t:.6f}", target, f"{src_fps:.6f}", src_w, src_h])
        k += 1
        if time.time() - last_log > 60:
            el = time.time() - t0
            _log(f"[extract] {clip}: {k} frames, {el:.0f}s, {k / el:.1f} fps")
            last_log = time.time()
    cap.release()

    _write_csv_atomic(csv_path, rows)
    el = time.time() - t0
    _log(f"[extract] {clip} ({split}): done, {len(rows)} frames ({encoded} newly encoded) in {el:.0f}s, {len(rows) / max(el, 1e-9):.1f} fps")
    return clip, split, len(rows), encoded, el


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("/media-src"), help="directory containing the *.MP4 clips")
    ap.add_argument("--dst", type=Path, default=Path("/datasets/HackEye"), help="dataset root")
    ap.add_argument("--fps", type=float, default=20, help="output frame rate")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--quality", type=int, default=100, help="JPEG quality (100 = max)")
    ap.add_argument("--val-clips", default="Chest-center-0002,Chest-center-0004",
                    help="comma list of clip stems that go to the val split; all others go to train")
    ap.add_argument("--clips", default=None, help="comma list of clip stems to process (smoke tests)")
    ap.add_argument("--workers", type=int, default=6, help="clips processed in parallel")
    ap.add_argument("--limit-seconds", type=float, default=None, help="only the first N seconds of each clip (smoke tests only)")
    args = ap.parse_args()

    val = {c for c in args.val_clips.split(",") if c}
    files = sorted(p for p in args.src.iterdir() if p.suffix.lower() == ".mp4")
    if args.clips:
        want = {c for c in args.clips.split(",") if c}
        missing = want - {p.stem for p in files}
        if missing:
            sys.exit(f"[extract] unknown clips: {sorted(missing)}")
        files = [p for p in files if p.stem in want]
    if not files:
        sys.exit(f"[extract] no clips found in {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)
    _log(f"[extract] {len(files)} clips -> {args.dst}, {args.fps} fps, {args.width}x{args.height}, q{args.quality}, val={sorted(val)}")

    totals = {"train": 0, "val": 0}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(files))) as pool:
        futs = {
            pool.submit(extract_clip, str(p), str(args.dst), "val" if p.stem in val else "train",
                        args.fps, args.width, args.height, args.quality, args.limit_seconds): p
            for p in files
        }
        for fut in as_completed(futs):
            clip, split, n, _, _ = fut.result()  # raises loudly on any clip failure
            totals[split] += n
    _log(f"[extract] finished in {time.time() - t0:.0f}s: train={totals['train']} val={totals['val']} total={sum(totals.values())}")


if __name__ == "__main__":
    main()
