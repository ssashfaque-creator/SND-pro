"""Paths and model knobs. Override with environment variables when needed."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("SNDINTEL_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


ROOT = project_root()
DATA_DIR = Path(os.environ.get("SNDINTEL_DATA_DIR", ROOT / "data")).resolve()
INCOMING_DIR = Path(os.environ.get("SNDINTEL_INCOMING", DATA_DIR / "incoming")).resolve()
PROCESSED_DIR = Path(os.environ.get("SNDINTEL_PROCESSED", DATA_DIR / "processed")).resolve()
SAMPLE_DIR = DATA_DIR / "sample"
MODEL_DIR = Path(os.environ.get("SNDINTEL_MODELS", DATA_DIR / "models")).resolve()
DB_PATH = Path(os.environ.get("SNDINTEL_DB", DATA_DIR / "warehouse.db")).resolve()

VOLUME_UNIT = os.environ.get("SNDINTEL_UOM", "MT")

# Insight / ML thresholds — tuned for monthly shop-SKU edible-oil volumes.
TRADE_LOAD_MULTIPLE = float(os.environ.get("SNDINTEL_TRADE_LOAD_X", "2.5"))
DROP_OFF_RATIO = float(os.environ.get("SNDINTEL_DROP_OFF_RATIO", "0.4"))
DIVERGENCE_GAP_PP = float(os.environ.get("SNDINTEL_DIVERGENCE_PP", "15"))
MIN_VOLUME_FLAG_MT = float(os.environ.get("SNDINTEL_MIN_FLAG_MT", "0.05"))
ISOLATION_CONTAMINATION = float(os.environ.get("SNDINTEL_IF_CONTAM", "0.06"))
FORECAST_MIN_PERIODS = int(os.environ.get("SNDINTEL_FORECAST_MIN_PERIODS", "8"))
CLUSTER_RANDOM_STATE = 42


def ensure_dirs() -> None:
    for path in (DATA_DIR, INCOMING_DIR, PROCESSED_DIR, SAMPLE_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
