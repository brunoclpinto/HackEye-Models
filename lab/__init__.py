"""Models Lab: web tool for browsing, correcting and re-mapping HackEye datasets."""

import sys
from pathlib import Path

__version__ = "0.1.0"

# The shared dataset code (datasets/common/*) is imported as top-level `common.*`, the way
# dataset_merge.py and pseudo_label.py import it. It is located relative to this package, not
# LAB_REPO (which only holds taxonomies/ and may be a fixture in tests).
_DATASETS = str(Path(__file__).resolve().parents[1] / "datasets")
if _DATASETS not in sys.path:
    sys.path.insert(0, _DATASETS)
