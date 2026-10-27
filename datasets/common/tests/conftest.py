import sys
from pathlib import Path

DATASETS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATASETS))
sys.path.insert(0, str(DATASETS / "vistas" / "conversion"))
