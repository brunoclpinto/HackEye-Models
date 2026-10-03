# GPU-only training rig for the Mapillary Vistas -> yolo26s-sem work. Same
# no-CPU-fallback policy as the main HackEye repo's RnD/Dockerfile, enforced
# by training/gpu_guard.py.
#
# Blackwell (RTX 50-series, sm_120) needs CUDA 12.8+. This base image ships it.
#
# Repo layout (see README.md): this Dockerfile lives in docker/, but the
# build context is the repo ROOT (set in docker/compose.yaml), so COPY paths
# below are relative to the repo root, not this file's own directory.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No fixed script -- this image runs the dataset converter/verifier under
# datasets/vistas/conversion/, and the trainer/evaluator under training/.
# Pick one via `docker compose run --rm train <path/to/script.py> ...`
# (see README.md for exact invocation paths).
ENTRYPOINT ["python"]
