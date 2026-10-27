"""Derives a merged-taxonomy dataset from an already-converted Vistas dataset
(the output of vistas_convert.py), collapsing classes per a merge map such as
taxonomies/HackEye26.1-48.yaml.

Thin wrapper: the logic lives in datasets/common/dataset_merge.py (which also
handles the HackEye master dataset). The taxonomy is built from Vistas'
config_v2.0.json; the output yaml keeps its historical name vistas-v2.0.yaml.

    python vistas_merge.py --src /datasets/vistas-yolo --dst /datasets/vistas-yolo-merge-v1 \\
        --map taxonomies/HackEye26.1-48.yaml --config /datasets/vistas-raw/config_v2.0.json
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.dataset_merge import run  # noqa: E402
from common.taxonomy import from_vistas_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="converted dataset root (output of vistas_convert.py)")
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--map", required=True, type=Path, help="merge map YAML, e.g. taxonomies/HackEye26.1-48.yaml")
    ap.add_argument("--config", required=True, type=Path, help="raw Vistas config_v2.0.json")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true", help="resolve the map and print it, write nothing")
    args = ap.parse_args()

    tax = from_vistas_config(args.config)
    ok = run(args.src, args.dst, args.map, tax, hashlib.sha256(args.config.read_bytes()).hexdigest(),
             args.splits.split(","), args.workers, args.dry_run,
             src_yaml_path=args.src / "vistas-v2.0.yaml", yaml_name="vistas-v2.0.yaml")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
