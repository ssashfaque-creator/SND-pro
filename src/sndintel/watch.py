"""Drop-folder watcher: any new CSV/XLSX in incoming/ re-runs the pipeline."""

from __future__ import annotations

import time
from pathlib import Path

from sndintel.config import INCOMING_DIR, SAMPLE_DIR, ensure_dirs
from sndintel.ingest.pipeline import run_pipeline

SALES_GLOBS = ("*.xlsx", "*.xlsm", "*.xls", "*.csv")
SHOP_HINTS = ("shop", "store", "outlet", "master", "universe")
TARGET_HINTS = ("target", "quota", "tgt")


def _is_target_file(path: Path) -> bool:
    name = path.name.lower()
    return any(h in name for h in TARGET_HINTS)


def _is_shop_file(path: Path) -> bool:
    name = path.name.lower()
    if _is_target_file(path):
        return False
    return any(h in name for h in SHOP_HINTS)


def latest_shop_file() -> Path | None:
    ensure_dirs()
    candidates = []
    for folder in (INCOMING_DIR, SAMPLE_DIR):
        if not folder.exists():
            continue
        for pat in SALES_GLOBS:
            for path in folder.glob(pat):
                if _is_shop_file(path):
                    candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def process_sales_file(path: Path, shop_path: Path | None = None) -> dict:
    shop = shop_path or latest_shop_file()
    return run_pipeline(path, shop_path=shop)


def scan_once() -> list[dict]:
    ensure_dirs()
    results = []
    shop = latest_shop_file()
    for pat in SALES_GLOBS:
        for path in sorted(INCOMING_DIR.glob(pat)):
            if _is_shop_file(path):
                continue
            marker = path.with_suffix(path.suffix + ".done")
            if marker.exists() and marker.stat().st_mtime >= path.stat().st_mtime:
                continue
            if _is_target_file(path):
                result = run_pipeline(shop_path=shop, targets_path=path)
            else:
                result = process_sales_file(path, shop)
            marker.write_text(str(result), encoding="utf-8")
            results.append(result)
    return results


def watch_forever(poll_seconds: float = 5.0) -> None:
    ensure_dirs()
    print(f"Watching {INCOMING_DIR} for sales files. Drop a CSV/XLSX to score it.")
    while True:
        try:
            done = scan_once()
            for item in done:
                print(f"Processed run {item.get('run_id')} period {item.get('latest_period')} insights={item.get('n_insights')}")
        except Exception as exc:  # noqa: BLE001 — watcher must stay alive
            print(f"Watch cycle failed: {exc}")
        time.sleep(poll_seconds)
