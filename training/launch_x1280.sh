#!/usr/bin/env bash
# 2026-10-18: build the 124-class 1280 train+val set, verify it, launch X-1280 (100 epochs) detached.
set -euo pipefail
cd "$(dirname "$0")/../docker"
R() { docker compose run --rm --entrypoint python train "$@"; }
R datasets/vistas/conversion/vistas_convert.py --src /datasets/vistas-raw --dst /datasets/vistas-yolo-1280 \
  --short-side 1280 --max-long-side 4096 --splits train,val --workers 12
R datasets/vistas/conversion/vistas_verify.py shapes --dst /datasets/vistas-yolo-1280
R datasets/vistas/conversion/vistas_verify.py yaml-check --yaml /datasets/vistas-yolo-1280/vistas-v2.0.yaml
docker compose run -d --name hackeye-x1280 train training/train_semantic.py \
  --model /base-models/yolo26x-sem-ade20k.pt --data /datasets/vistas-yolo-1280/vistas-v2.0.yaml \
  --imgsz 1280 --batch 1 --cls-pw 1.0 --workers 5 --no-plots --epochs 100 --close-mosaic 3 \
  --save-period 1 --name vistas124-x-1280
echo LAUNCHED
