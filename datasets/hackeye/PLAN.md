# Plan: the HackEye frames dataset (pseudo-labelled by the X model)

Status: **in progress**, started 2026-10-26. Results and the final state go in `PROGRESS.md`.
How to reach and use the finished dataset is in [README.md](README.md) (same folder).

## Goal

Turn the whole HackEye media library (`~/media/hackeye/*.MP4`, 19 GoPro clips, about 2.8 h) into
a YOLO semantic-segmentation dataset:

- **Frames:** 1280×720 JPG at 20 fps, maximum quality.
- **Masks:** one 124-class Vistas mask per frame, predicted by the dataset-creation model
  `models/Yolo26x-sem-Vistas124-HackEye-1280.pt`.

The masks are pseudo-labels. They are the starting point for curation, which comes next: the
retrained app model will learn from Vistas plus curated HackEye frames.

## Fixed decisions

| Decision | Choice | Why |
|---|---|---|
| Location | `~/media/hackeye-models/datasets/HackEye26.1-124/` (`/datasets/HackEye26.1-124` in the container) | It's too big for git. It sits next to `vistas-raw`, following the central media rule; the compose file already mounts it. |
| Layout | `images/{train,val}/<stem>.jpg` + `masks/{train,val}/<stem>.png` + `HackEye26.1-124.yaml` | This is the same SemanticDataset layout as the Vistas conversion (`masks_dir: masks`), so `train_semantic.py --data /datasets/HackEye26.1-124/HackEye26.1-124.yaml` works unchanged. |
| Stem | `<clip>_<k:06d>`, e.g. `Chest-center-0004_000123`; `k` is the 20 fps index, so `t = k / 20` s | It sorts in time, and any frame maps straight back to a timestamp in its clip. |
| Frame pick | For `t = k/20`, take source frame `round(t × fps)` until the clip ends | 59.94 fps sources then keep every 3rd frame. 23.976 fps sources drop about 1 frame in 6, unevenly. |
| Resize | Area resize (`cv2.INTER_AREA`) to exactly 1280×720 | All clips are 16:9 (2704×1520 is 1.779, so a 0.05% stretch). |
| JPEG | OpenCV quality 100 with optimized Huffman tables, 4:2:0 | This is the maximum quality setting: about 590 KB a frame, about 113 GB in total. |
| Masks | 8-bit palette PNG holding class ids 0..123 with the Vistas colours | This matches the Vistas conversion: PIL reads ids, and viewers show colours. |
| Split | **by clip**: `val` = `Chest-center-0002`, `Chest-center-0004`; `train` = the other 17 | These two clips are the Lab's test and review footage. Keeping them out of `train` means a model trained here never sees the footage the app is judged on. Note: `Chest-center-0006` is the next GoPro chapter after 0002, so their first and last seconds are adjacent. |
| Inference scale | **Multi-scale: mean softmax over short sides 736 and 1280** (net inputs 1312×736 and 2304×1280) | Step 0 result, below. It runs at about 4 frames/s at batch 1, which is faster than batch 4 or 8, so the full set takes about 13.5 h. |
| Confidence | Per-frame stats in `frames.csv` (mean max-softmax, 10th percentile, share of pixels below 0.5) | These let curation sort frames by how unsure the teacher is. Full confidence maps would add about 30 GB, so they aren't stored. |

## Steps

0. **Scale check.** Run the X model on a 500-image Vistas val subset at short side 736 (native
   frames) and at 1280 (training scale), comparing mIoU and time. The faster scale wins unless the
   larger one is clearly better (more than 0.01 mIoU). Both outputs are scratch and get purged.

   **Result (2026-10-26).** The "1280" condition used real 720-detail images upscaled: Vistas
   shrunk to 736 short side, then resized back. That matches what HackEye frames give the net.

   | Short side | mIoU (124) | Pixel acc | Road | Sidewalk | Bike lane | Crosswalk | Pedestrian area | Curb | Curb cut |
   |---|---|---|---|---|---|---|---|---|---|
   | 736 | 0.402 | 0.896 | 0.851 | 0.644 | 0.466 | 0.509 | **0.586** | 0.589 | 0.245 |
   | 1280 (upscaled) | **0.424** | 0.903 | **0.865** | **0.652** | 0.437 | 0.466 | 0.499 | **0.618** | 0.259 |
   | **736 + 1280 mean** | 0.422 | **0.904** | 0.864 | 0.645 | 0.463 | **0.517** | 0.546 | 0.615 | **0.263** |

   On its own, 1280 loses on bike lane, crosswalk and pedestrian area, which are broad
   navigable-surface classes (the project's priority). The two-scale mean recovers most of
   that, keeps the mIoU, curb and marking gains, and has the best pixel accuracy. On real
   1280-detail images, 1280 scored 0.430 mIoU, so upscaling costs only about 0.006.
1. **Extract** (`datasets/hackeye/extract_frames.py`, CPU, in the train container). The script is
   resumable and writes atomically. It produces `frames.csv` rows: stem, split, clip, k, t_sec,
   src_frame, src_fps.
2. **Pseudo-label** (`datasets/hackeye/pseudo_label.py`, GPU, launched detached). The script is
   resumable and skips existing masks. It writes the masks, the confidence columns and per-class
   pixel totals.
3. **Finalize.** Write `HackEye26.1-124.yaml` (124 Vistas names + `label_mapping` voids, like
   `vistas-v2.0.yaml`) and `dataset_manifest.json`. The manifest records the model file and sha256,
   the repo commit, all parameters, a per-clip table, counts and the class pixel histogram.
4. **Verify.**
   - Counts: frames equal masks equal csv rows, and match the expected count per clip.
   - Shapes: every frame is 1280×720.
   - Masks open as `P` mode.
   - Spot-check colour overlays across clips and times by eye.
   - `vistas_verify.py yaml-check` passes on `HackEye26.1-124.yaml`.
5. **Copy to Drive: deferred (user, 2026-10-26).** The dataset is curated first, and the copy
   happens only on the user's go. When it does: build an uncompressed tar of the curated dataset
   root (JPGs don't compress), then `rclone copy` it to `bpbroasGDrive:works/HackEye/datasets/`
   (next to `vistasV2Raw-jxl.zip`). rclone checks the md5. Then delete the local tar.
6. **Purge.** Remove the scale-check data and outputs, logs, exited containers,
   `__pycache__`, and scratch files. The **kept** items are the dataset, the two scripts and the docs.
7. **Docs.** Add `datasets/hackeye/README.md` (where the dataset lives, its layout, the Drive copy,
   how to rebuild it), and update README and PROGRESS. Commit, push, and email the user.

## Worker recommendation

**Sonnet** writes `extract_frames.py` and `pseudo_label.py` from the spec below. They are two
self-contained scripts with a clear contract, and they reuse `vistas_config.load_taxonomy` and the
predictor pattern from `diagnose_surfaces.py`. Opus does the following:

- runs the scale check and picks the scale;
- reviews the scripts;
- runs and verifies the long jobs;
- handles the purge, the docs and git (the Drive copy waits for curation).

## Not in scope

- The curation itself: human review of the masks and frame selection.
- Retraining.
- Any change to the HackEye app or the Lab.
