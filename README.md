# HackEye-Models
YOLO training for the HackEye app.

## Current focus: Vistas semantic segmentation

Training `yolo26s-sem` on Mapillary Vistas v2.0 (124 classes) to add curb/
curb-cut detection, which the app's current 19-class Cityscapes model has no
concept of. See **[PLAN.md](PLAN.md)** for the design and **[PROGRESS.md](PROGRESS.md)**
for current status and the exact next step — read PROGRESS.md first.

Training priority for this pass: broad navigable-surface classes first —
`road`, `sidewalk`, `bike-lane`, `pedestrian-area`, `crosswalk` (plain +
zebra), `parking`/`parking-aisle`, `road-shoulder`, `service-lane`,
`rail-track`, `traffic-island`. `curb`/`curb-cut` are deliberately deferred
to a later fine-tuning pass — they're a tiny fraction of pixels per image and
need far more iterations to get right than the broad classes do. See
`datasets/vistas/conversion/vistas_classes.txt` for the full 124-class list.

## Models

Our trained models live in `models/` (only `*HackEye*` files are tracked):

| File | Use |
|---|---|
| `Yolo26m-sem-HackEye26.1-48-1280.pt` | **App release** for HackEye v26.1: m, 48-class merged taxonomy, trained at 1280 |
| `Yolo26x-sem-Vistas124-HackEye-1280.pt` | **Dataset creation**: x, full 124-class Vistas, trained at 1280. It's the base model for curating and pre-labelling training images; it isn't shipped in the app. |

App releases are named `Yolo26<size>-sem-HackEye<app version>-<classes>-<train res>.pt`. Dataset-creation
models are named `Yolo26<size>-sem-<dataset><classes>-HackEye-<train res>.pt`. The app release's class merge map carries the same name:
`datasets/vistas/conversion/merges/HackEye26.1-48.yaml`. See **PROGRESS.md** for how
each was trained and evaluated.

## Setup

Docker-based; no Jupyter notebooks. From `docker/` (or pass
`-f docker/compose.yaml` from the repo root):

```bash
cd docker
docker compose build
docker compose run --rm train <path/to/script>.py ...
# e.g. training/train_semantic.py, training/eval_semantic.py,
#      datasets/vistas/conversion/vistas_convert.py, .../vistas_verify.py
```

The container mounts the repo root at `/app` (the build context is the repo
root even though the Dockerfile/compose file live in `docker/`), so script
paths are relative to the repo root, not to `docker/`. Downloaded base
checkpoints (official `yolo26*` assets) live in the shared `~/models/ultralytics`
dir, mounted read-only at `/base-models`; `models/` holds only our own HackEye
fine-tunes. See `docker/Dockerfile` /
`docker/compose.yaml` and **PROGRESS.md** for exact commands, volume
mounts, and environment (single RTX 5060 Ti, 16GB VRAM).

## Datasets

### Mapillary Vistas v2.0 (current)

124-class street-scene segmentation dataset — chosen because it natively
labels `curb` and `curb cut` (Cityscapes' 19-class taxonomy has no curb
concept at all), alongside `pedestrian area`, `bike lane`, and other classes
relevant to navigation. Requires registering and accepting Mapillary's
research license at <https://www.mapillary.com/dataset/vistas>. See
**PLAN.md** for the full design and **PROGRESS.md** for dataset paths,
conversion status, and verification results.

### HackEye frames: HackEye26.1-124 (pseudo-labelled, curated)

The HackEye video library: 45,157 frames (train 14,830, curated from 169,696; val 30,327), about 23 GB.

- **Frames:** 1280×720 JPG at 20 fps.
- **Masks:** 124-class pseudo-labels from the X dataset-creation model.
- **Split:** by clip, and the Lab's test clips are kept in `val`.

The dataset is too big for git. It lives at `~/media/hackeye-models/datasets/HackEye26.1-124/`, which is
`/datasets/HackEye26.1-124/HackEye26.1-124.yaml` in the container. It is copied to Google Drive only
on the user's go. See **[datasets/hackeye/README.md](datasets/hackeye/README.md)** for the location,
layout and rebuild commands.

### Previously explored

- **Cityscapes** — the original plan was to use Cityscapes for its polish
  and community adoption. Cityscapes declined to grant dataset access, so
  this was abandoned in favor of Mapillary Vistas (which also happens to be
  the better fit anyway, given the curb/curb-cut labels above).
- **Bus / OpenImagesV7** — an earlier, separate effort to build a bus-detection
  dataset by manually scraping OpenImagesV7's web visualizer. Abandoned when
  the project's focus shifted to Vistas-based semantic segmentation; the raw
  scraped HTML has been removed from the repo.

## Licences and thanks
The scripts in this repository are [Apache-2.0](LICENSE). **The trained models in `models/` are not:**
- They are fine-tuned from Ultralytics YOLO26 weights, which makes them **AGPL-3.0**.
- They are trained on **Mapillary Vistas** (CC BY-NC-SA 4.0, research licence), warm-started from ADE20K-trained weights.

The models are free for non-commercial use, with attribution. See
**[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)** for the details and the acknowledgements.

