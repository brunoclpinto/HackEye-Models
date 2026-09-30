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
`vistas_classes.txt` for the full 124-class list.

## Setup

Docker-based; no Jupyter notebooks. From this repo's root:

```bash
docker compose build
docker compose run --rm train <script>.py ...   # vistas_convert.py, vistas_verify.py, train_semantic.py, eval_semantic.py
```

See `Dockerfile` / `compose.yaml` and **PROGRESS.md** for exact commands,
volume mounts, and environment (single RTX 5060 Ti, 16GB VRAM).

## Datasets

### Mapillary Vistas v2.0 (current)

124-class street-scene segmentation dataset — chosen because it natively
labels `curb` and `curb cut` (Cityscapes' 19-class taxonomy has no curb
concept at all), alongside `pedestrian area`, `bike lane`, and other classes
relevant to navigation. Requires registering and accepting Mapillary's
research license at <https://www.mapillary.com/dataset/vistas>. See
**PLAN.md** for the full design and **PROGRESS.md** for dataset paths,
conversion status, and verification results.

### Previously explored

- **Cityscapes** — the original plan was to use Cityscapes for its polish
  and community adoption. Cityscapes declined to grant dataset access, so
  this was abandoned in favor of Mapillary Vistas (which also happens to be
  the better fit anyway, given the curb/curb-cut labels above).
- **Bus / OpenImagesV7** — an earlier, separate effort to build a bus-detection
  dataset by manually scraping OpenImagesV7's web visualizer. Abandoned when
  the project's focus shifted to Vistas-based semantic segmentation; the raw
  scraped HTML has been removed from the repo.
