# GPU-only training rig for the Mapillary Vistas -> yolo26s-sem work. Same
# no-CPU-fallback policy as RnD/Dockerfile, enforced by gpu_guard.py.
#
# Blackwell (RTX 50-series, sm_120) needs CUDA 12.8+. This base image ships it
# -- same base as RnD/Dockerfile, but built directly FROM it (not FROM
# hackeye-rnd) so training/ has no build-order dependency on RnD/ and can be
# lifted into its own repo without carrying that coupling.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /training

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No fixed script -- this image runs vistas_convert.py, vistas_verify.py,
# train_semantic.py, and eval_semantic.py at different points in the
# pipeline. Pick one via `docker compose run --rm train <script.py> ...`.
ENTRYPOINT ["python"]
