"""Mask IO: P-mode PNG <-> uint8 id arrays, upload validation, single-channel PNG encoding."""

from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

W, H = 1280, 720


class MaskError(ValueError):
    """Invalid mask upload (mapped to HTTP 422)."""


def read_mask(path: Path) -> np.ndarray:
    """(H, W) uint8 array of base ids from a P- or L-mode PNG."""
    with Image.open(path) as im:
        if im.mode not in ("P", "L"):
            raise MaskError(f"{path.name}: mask mode {im.mode!r}, expected P or L")
        return np.array(im, dtype=np.uint8)


def write_mask(path: Path, arr: np.ndarray, palette: list[int]) -> None:
    """Atomic P-mode PNG write (tmp + os.replace) with a 768-entry palette."""
    path = Path(path)
    h, w = arr.shape
    im = Image.frombytes("P", (w, h), np.ascontiguousarray(arr, dtype=np.uint8).tobytes())
    im.putpalette(palette)
    tmp = path.with_name(path.name + ".tmp")
    im.save(tmp, format="PNG")
    os.replace(tmp, path)


def encode_gray_png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(arr, dtype=np.uint8))
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def decode_upload(data: bytes, valid_ids: set[int]) -> np.ndarray:
    """Validates a PUT body: PNG, 1280x720, grayscale or RGB(A) with R=G=B, ids within the taxonomy (or 255)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            if im.format != "PNG":
                raise MaskError("body must be a PNG")
            if im.size != (W, H):
                raise MaskError(f"mask must be {W}x{H}, got {im.size[0]}x{im.size[1]}")
            if im.mode in ("L", "P"):
                arr = np.array(im, dtype=np.uint8)
            elif im.mode in ("RGB", "RGBA"):
                a = np.array(im, dtype=np.uint8)
                if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
                    raise MaskError("RGB channels differ (need R=G=B)")
                arr = np.ascontiguousarray(a[..., 0])
            else:
                raise MaskError(f"unsupported PNG mode {im.mode!r}")
    except MaskError:
        raise
    except Exception as e:  # noqa: BLE001
        raise MaskError(f"cannot decode PNG: {e}") from e
    present = np.flatnonzero(np.bincount(arr.ravel(), minlength=256))
    bad = [int(i) for i in present if int(i) not in valid_ids and int(i) != 255]
    if bad:
        raise MaskError(f"ids not in the taxonomy: {bad[:10]}")
    return arr


def count_px(arr: np.ndarray) -> np.ndarray:
    return np.bincount(arr.ravel(), minlength=256).astype(np.int64)


def encode_counts(counts: np.ndarray) -> str:
    return ";".join(f"{int(i)}:{int(counts[i])}" for i in np.flatnonzero(counts))


def decode_counts(s: str) -> dict[int, int]:
    out: dict[int, int] = {}
    if s:
        for part in s.split(";"):
            i, c = part.split(":")
            out[int(i)] = int(c)
    return out
