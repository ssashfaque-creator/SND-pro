"""Install or replace app code from a ZIP. Never touches the warehouse."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

SKIP_NAMES = {".venv", ".git", "data", "__pycache__", ".pytest_cache", "warehouse.db"}


def find_download_zip(downloads: Path | None = None) -> Path | None:
    folder = Path(downloads or Path.home() / "Downloads")
    if not folder.is_dir():
        return None
    zips = sorted(folder.glob("SND-pro*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def apply_code_zip(archive: Path, dest: Path) -> dict[str, str]:
    """Unpack a GitHub/browser ZIP over dest. Leaves .venv and the warehouse alone."""
    archive = Path(archive).expanduser().resolve()
    dest = Path(dest).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"No ZIP at {archive}")
    dest.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="sndintel-upd-"))
    try:
        shutil.unpack_archive(str(archive), tmp)
        src = _find_root(tmp)
        _copy_tree(src, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"src_zip": str(archive), "app_dir": str(dest)}


def _find_root(unpacked: Path) -> Path:
    hits = [p for p in unpacked.iterdir() if p.is_dir() and p.name.startswith("SND-pro")]
    if hits:
        return hits[0]
    if (unpacked / "pyproject.toml").exists() or (unpacked / "src").exists():
        return unpacked
    nested = list(unpacked.glob("*/pyproject.toml"))
    if nested:
        return nested[0].parent
    raise FileNotFoundError("ZIP did not contain the SND Intelligence app folder.")


def _copy_tree(src: Path, dest: Path) -> None:
    for item in src.iterdir():
        if item.name in SKIP_NAMES:
            continue
        target = dest / item.name
        if item.is_dir():
            if target.exists() and target.is_dir():
                shutil.rmtree(target)
            shutil.copytree(item, target, ignore=shutil.ignore_patterns(*SKIP_NAMES, "*.pyc"))
        else:
            shutil.copy2(item, target)
