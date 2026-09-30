"""GPU is required, never optional -- same policy as RnD/src/gpu_guard.py.

Copied rather than imported: training/ is meant to be self-contained (see the
plan doc) so it can be extracted into its own repo later without carrying an
inward dependency on RnD/src. Keep the two in sync by hand if the shared
policy changes; there are only ~15 lines of actual logic here.

Adds one thing RnD/src/gpu_guard.py doesn't need: a VRAM floor check. The
inference rig never came close to its budget; a semantic-segmentation
training run on a 16GB card can, and OOM'ing 40 minutes into a run is a much
worse failure mode than refusing to start.
"""

import sys

import torch


def assert_torch_cuda() -> torch.device:
    """Exit the process if CUDA is unavailable. Returns the device to use."""
    if not torch.cuda.is_available():
        _fail(
            "CUDA is not available to PyTorch inside this container.\n"
            "This rig has no CPU fallback by design (see training/gpu_guard.py).\n"
            "Check: `docker compose run --rm train nvidia-smi`, and that the\n"
            "`nvidia` container runtime is selected (see training/compose.yaml)."
        )

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"[gpu_guard] CUDA device: {name}, compute capability {capability}", file=sys.stderr)

    if capability < (7, 0):
        _fail(
            f"CUDA device capability {capability} looks too old for this pipeline. "
            "Refusing to continue rather than run inference at a crawl."
        )

    return torch.device("cuda:0")


def assert_min_vram(min_gb: float) -> float:
    """Exit if the GPU's total VRAM is below min_gb. Returns the actual GB.

    This is a floor check against the card's total memory, not a prediction
    of whether a specific batch/imgsz will fit -- it only catches "wrong GPU
    entirely" (e.g. accidentally running against an 8GB card), not a
    marginal OOM at batch=32. See the plan doc's VRAM table for per-config
    estimates; those still need to be probed empirically.
    """
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[gpu_guard] VRAM: {total_gb:.1f} GB total", file=sys.stderr)
    if total_gb < min_gb:
        _fail(
            f"GPU has {total_gb:.1f} GB VRAM, need at least {min_gb} GB for this "
            "training config. Lower --batch/--imgsz or pass a smaller --min-vram-gb "
            "if you've already accounted for this."
        )
    return total_gb


def assert_onnxruntime_cuda(session) -> None:
    """Verify an onnxruntime.InferenceSession is actually using CUDA.

    ONNXRuntime does not raise when CUDAExecutionProvider fails to load — it
    just falls back to CPUExecutionProvider. `providers=[...]` on session
    creation is a *request*, not a guarantee. This checks what was actually
    granted.
    """
    active = session.get_providers()
    if "CUDAExecutionProvider" not in active:
        _fail(
            f"onnxruntime session is NOT using CUDA (active providers: {active}). "
            "It silently fell back to CPU. Check the onnxruntime-gpu install "
            "and CUDA/cuDNN versions inside the container."
        )
    print(f"[gpu_guard] onnxruntime providers: {active}", file=sys.stderr)


def _fail(message: str) -> None:
    print(f"[gpu_guard] FATAL: {message}", file=sys.stderr)
    sys.exit(1)
