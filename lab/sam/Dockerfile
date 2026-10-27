# SAM2 point/box service for the Models Lab. Recipe follows ~/media-gen/segmentation/Dockerfile.
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY app.py .

# runs as the host user (compose); torch/hydra want a writable HOME
ENV HOME=/tmp USER=lab LOGNAME=lab
EXPOSE 8000
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
