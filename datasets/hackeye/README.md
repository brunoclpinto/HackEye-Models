# HackEye frames dataset

The whole HackEye video library as a YOLO semantic-segmentation dataset:

- **Frames:** 1280×720 JPG at 20 fps, maximum quality.
- **Masks:** 124-class Vistas pseudo-labels from the dataset-creation model
  `models/Yolo26x-sem-Vistas124-HackEye-1280.pt`.

The masks are the **teacher's predictions, not ground truth**. Curate them before training on them.
The design and the scale study are in [PLAN.md](PLAN.md); the build record is in `PROGRESS.md`.
The curation (near-duplicate trimming) is in [CURATION.md](CURATION.md).

## Where it is

The dataset is too big for git, so it lives outside the repo, in the shared media dir:

| What | Host | In the train container |
|---|---|---|
| Dataset root | `~/media/hackeye-models/datasets/HackEye/` | `/datasets/HackEye` |
| Data yaml | `.../HackEye/HackEye.yaml` | `/datasets/HackEye/HackEye.yaml` |
| Source videos (read-only) | `~/media/hackeye/*.MP4` | `/media-src` |
| Drive copy | **None yet.** The tar goes to `bpbroasGDrive:works/HackEye/datasets/HackEye.tar` only after curation, on the user's go. | n/a |

`docker/compose.yaml` mounts all three through `HACKEYE_DATASETS_DIR` and `HACKEYE_MEDIA_DIR`.
Both have defaults, so nothing needs setting on this host. Use it like any other data yaml:

```bash
cd docker
docker compose run --rm train training/train_semantic.py --data /datasets/HackEye/HackEye.yaml ...
```

## Layout

```
HackEye/
  HackEye.yaml              124 Vistas names, label_mapping voids, masks_dir: masks, train/val
  dataset_manifest.json     model + sha256, scales, extraction params, per-clip table, counts, class pixels
  frames.csv                one row per frame: stem, split, clip, k, t_sec, src_frame, src_fps, src_w, src_h,
                            conf_mean, conf_p10, lowconf_frac
  images/{train,val}/<clip>_<k:06d>.jpg   1280x720, OpenCV q100 + optimized Huffman
  masks/{train,val}/<clip>_<k:06d>.png    8-bit palette PNG, pixel value = Vistas class id 0..123
  meta/                     build bookkeeping: per-clip frame csvs, per-split confidence and class-pixel csvs, class_pixels.json
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
| train | 17 | 169,696 |
| val | 2 | 30,327 |
| **total** | 19 | **200,023** (about 2.8 h of video) |

## Rebuilding it

Both scripts are resumable and skip work that is already done. They can be run again after new
clips land in `~/media/hackeye/`, but the existing train/val split rule stays.

```bash
cd docker
# 1. frames (CPU, ~20 min for the 19 clips)
docker compose run --rm -T --entrypoint python train datasets/hackeye/extract_frames.py
# 2. masks + finalize (GPU, ~11 h at ~5 frames/s; run it detached with setsid nohup)
docker compose run --rm -T --entrypoint python train datasets/hackeye/pseudo_label.py --imgsz 736,1280 --batch 1
# finalize only (frames.csv, HackEye.yaml, dataset_manifest.json)
docker compose run --rm -T --entrypoint python train datasets/hackeye/pseudo_label.py --finalize-only
```

**Run with `-T` when detached.** Without it, a detached `compose run` holds its log output back.

**Restoring from Drive** (once the curated copy exists):

```bash
rclone copy bpbroasGDrive:works/HackEye/datasets/HackEye.tar ~/media/hackeye-models/datasets/
tar -xf ~/media/hackeye-models/datasets/HackEye.tar -C ~/media/hackeye-models/datasets/
rm ~/media/hackeye-models/datasets/HackEye.tar
```
