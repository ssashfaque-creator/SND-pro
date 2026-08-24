"""Paths and model knobs. Override with environment variables when needed."""

from __future__ import annotations

import os
import shutil
import sys
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


def _under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def user_data_dir() -> Path:
    """Durable warehouse location. App updates must not wipe history.

    Mac: ~/Library/Application Support/SND Intelligence
    Linux: ~/.local/share/sndintel
    Tests / SNDINTEL_USE_REPO_DATA=1: <repo>/data
    """
    env = os.environ.get("SNDINTEL_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    if os.environ.get("SNDINTEL_USE_REPO_DATA", "").lower() in {"1", "true", "yes"}:
        return (project_root() / "data").resolve()
    if _under_pytest():
        return (project_root() / "data").resolve()
    home = Path.home()
    if sys.platform == "darwin":
        return (home / "Library" / "Application Support" / "SND Intelligence").resolve()
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return (Path(xdg) / "sndintel").resolve()
    return (home / ".local" / "share" / "sndintel").resolve()


ROOT = project_root()
DATA_DIR = user_data_dir()
INCOMING_DIR = Path(os.environ.get("SNDINTEL_INCOMING", DATA_DIR / "incoming")).resolve()
PROCESSED_DIR = Path(os.environ.get("SNDINTEL_PROCESSED", DATA_DIR / "processed")).resolve()
SAMPLE_DIR = DATA_DIR / "sample"
MODEL_DIR = Path(os.environ.get("SNDINTEL_MODELS", DATA_DIR / "models")).resolve()
MASTER_DIR = DATA_DIR / "masters"
DB_PATH = Path(os.environ.get("SNDINTEL_DB", DATA_DIR / "warehouse.db")).resolve()

VOLUME_UNIT = os.environ.get("SNDINTEL_UOM", "MT")

# Insight / ML thresholds — tuned for monthly shop-SKU edible-oil volumes.
TRADE_LOAD_MULTIPLE = float(os.environ.get("SNDINTEL_TRADE_LOAD_X", "2.5"))
DROP_OFF_RATIO = float(os.environ.get("SNDINTEL_DROP_OFF_RATIO", "0.4"))
DIVERGENCE_GAP_PP = float(os.environ.get("SNDINTEL_DIVERGENCE_PP", "15"))
MIN_VOLUME_FLAG_MT = float(os.environ.get("SNDINTEL_MIN_FLAG_MT", "0.05"))
MIN_MATERIAL_MT = float(os.environ.get("SNDINTEL_MIN_MATERIAL_MT", "0.05"))
CORE_VOLUME_SHARE = float(os.environ.get("SNDINTEL_CORE_SHARE", "0.80"))
MIDDLE_VOLUME_SHARE = float(os.environ.get("SNDINTEL_MIDDLE_SHARE", "0.95"))
OCCASIONAL_BILLED_RATE = float(os.environ.get("SNDINTEL_OCCASIONAL_RATE", "0.35"))
ISOLATION_CONTAMINATION = float(os.environ.get("SNDINTEL_IF_CONTAM", "0.06"))
FORECAST_MIN_PERIODS = int(os.environ.get("SNDINTEL_FORECAST_MIN_PERIODS", "8"))
CLUSTER_RANDOM_STATE = 42
YOY_MIN_LY_MT = float(os.environ.get("SNDINTEL_YOY_MIN_LY_MT", "0.5"))
WHALE_AMS_MT = float(os.environ.get("SNDINTEL_WHALE_AMS_MT", "1.0"))
CALLS_PER_DAY = float(os.environ.get("SNDINTEL_CALLS_PER_DAY", "25"))
DSR_DAY_CAP = int(os.environ.get("SNDINTEL_DSR_DAY_CAP", "12"))
SPAN_OVERLOAD = float(os.environ.get("SNDINTEL_SPAN_OVERLOAD", "1.2"))
VISIT_SUSPECT_RATE = float(os.environ.get("SNDINTEL_VISIT_SUSPECT_RATE", "0.95"))
VISIT_SUSPECT_UNIVERSE = int(os.environ.get("SNDINTEL_VISIT_SUSPECT_UNI", "400"))
EXPECTED_FORMULA = os.environ.get("SNDINTEL_EXPECTED_FORMULA", "expected_v3 · paced national day curve")
APP_BRANCH = os.environ.get("SNDINTEL_APP_BRANCH", "cursor/actionable-ops-layer-2f34")
GITHUB_REPO = os.environ.get("SNDINTEL_GITHUB_REPO", "ssashfaque-creator/SND-pro")
APP_DIR = Path(os.environ.get("SNDINTEL_APP_DIR", str(Path.home() / "sndintel"))).expanduser()


def migrate_legacy_warehouse() -> None:
    """Copy a repo-local warehouse into the user data dir once, if the user dir is empty."""
    dest = Path(os.environ.get("SNDINTEL_DB", DATA_DIR / "warehouse.db")).resolve()
    if dest.exists():
        return
    legacy = ROOT / "data" / "warehouse.db"
    if not legacy.exists() or legacy.resolve() == dest:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy, dest)


def ensure_dirs() -> None:
    migrate_legacy_warehouse()
    for path in (DATA_DIR, INCOMING_DIR, PROCESSED_DIR, SAMPLE_DIR, MODEL_DIR, MASTER_DIR):
        path.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
