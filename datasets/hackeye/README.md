# HackEye frames dataset (HackEye26.1-124)

The HackEye video library as a YOLO semantic-segmentation dataset, named `HackEye26.1-124`
(HackEye app version 26.1, 124 Vistas classes). Train is curated: near-duplicate frames were removed
(config `fps2_sim0.90`, see [CURATION.md](CURATION.md)). Val is complete.

- **Frames:** 1280×720 JPG at 20 fps, maximum quality.
- **Masks:** 124-class Vistas pseudo-labels from the dataset-creation model
  `models/Yolo26x-sem-Vistas124-HackEye-1280.pt`.

The masks are the **teacher's predictions, not ground truth**. Known teacher error: "nature snow" is breaking
surf, sea foam and sun glare on the seafront clips.
The design and the scale study are in [PLAN.md](PLAN.md); the build record is in `PROGRESS.md`.
The curation (near-duplicate trimming) is in [CURATION.md](CURATION.md).

## Where it is

The dataset is too big for git, so it lives outside the repo, in the shared media dir:

| What | Host | In the train container |
|---|---|---|
| Dataset root | `~/media/hackeye-models/datasets/HackEye26.1-124/` | `/datasets/HackEye26.1-124` |
| Data yaml | `.../HackEye26.1-124/HackEye26.1-124.yaml` | `/datasets/HackEye26.1-124/HackEye26.1-124.yaml` |
| Source videos (read-only) | `~/media/hackeye/*.MP4` | `/media-src` |
| Drive copy | **None yet.** The tar goes to `bpbroasGDrive:works/HackEye/datasets/HackEye26.1-124.tar` only after curation, on the user's go. | n/a |

`docker/compose.yaml` mounts all three through `HACKEYE_DATASETS_DIR` and `HACKEYE_MEDIA_DIR`.
Both have defaults, so nothing needs setting on this host. Use it like any other data yaml:

```bash
cd docker
docker compose run --rm train training/train_semantic.py --data /datasets/HackEye26.1-124/HackEye26.1-124.yaml ...
```

## Layout

```
HackEye26.1-124/
  HackEye26.1-124.yaml              124 Vistas names, label_mapping voids, masks_dir: masks, train/val
  dataset_manifest.json     model + sha256, scales, extraction params, per-clip table, counts, class pixels
  frames.csv                one row per frame: stem, split, clip, k, t_sec, src_frame, src_fps, src_w, src_h,
                            conf_mean, conf_p10, lowconf_frac
  images/{train,val}/<clip>_<k:06d>.jpg   1280x720, OpenCV q100 + optimized Huffman
  masks/{train,val}/<clip>_<k:06d>.png    8-bit palette PNG, pixel value = Vistas class id 0..123
  meta/                     build bookkeeping: per-clip frame csvs, per-split confidence and class-pixel csvs, class_pixels.json,
                            curation.json (config, params, before/kept/deleted counts)
  curation/                 report.md + report.json of the dry-run sweep, fps2_sim0.90/ keep, group and hard-review lists
```

- **Time:** `k` is the 20 fps index. The frame shows time `t = k / 20` s of `<clip>.MP4` (source
  frame `src_frame`).
- **Split:** by clip. `val` is `Chest-center-0002` and `Chest-center-0004`, the Lab's test and
  review footage, so a model trained on `train` never sees what the app is judged on. `train` is
  the other 17 clips.
- **Confidence:** these columns rank frames for curation, with the lowest `conf_mean` or highest
  `lowconf_frac` first. They come from the mean softmax of two scales, short sides 736 and 1280.

## Counts

| Split | Clips | Frames |
|---|---|---|
| train | 17 | 14,830 (curated from 169,696) |
| val | 2 | 30,327 (complete) |
| **total** | 19 | **45,157**, about 23 GB (200,023 frames, 2.8 h of video, before curation) |

## Rebuilding it

Both scripts are resumable and skip work that is already done. They can be run again after new
clips land in `~/media/hackeye/`, but the existing train/val split rule stays.

```bash
cd docker
# 1. frames (CPU, ~20 min for the 19 clips)
docker compose run --rm -T --entrypoint python train datasets/hackeye/extract_frames.py
# 2. masks + finalize (GPU, ~11 h at ~5 frames/s; run it detached with setsid nohup)
docker compose run --rm -T --entrypoint python train datasets/hackeye/pseudo_label.py --imgsz 736,1280 --batch 1
# finalize only (frames.csv, HackEye26.1-124.yaml, dataset_manifest.json)
docker compose run --rm -T --entrypoint python train datasets/hackeye/pseudo_label.py --finalize-only
# 3. curation: dry-run sweep (CPU, ~1 min), then keep only the picked config's train frames and re-finalize
docker compose run --rm -T --entrypoint python -e CUDA_VISIBLE_DEVICES= train datasets/hackeye/curate.py --split train
docker compose run --rm -T --entrypoint python -e CUDA_VISIBLE_DEVICES= train datasets/hackeye/apply_curation.py --config fps2_sim0.90 --apply
docker compose run --rm -T --entrypoint python train datasets/hackeye/pseudo_label.py --finalize-only
```

Step 3 deletes the dropped train frames. Without `--apply`, `apply_curation.py` is a dry run.

**Run with `-T` when detached.** Without it, a detached `compose run` holds its log output back.

**Restoring from Drive** (once the curated copy exists):

```bash
rclone copy bpbroasGDrive:works/HackEye/datasets/HackEye26.1-124.tar ~/media/hackeye-models/datasets/
tar -xf ~/media/hackeye-models/datasets/HackEye26.1-124.tar -C ~/media/hackeye-models/datasets/
rm ~/media/hackeye-models/datasets/HackEye26.1-124.tar
```
