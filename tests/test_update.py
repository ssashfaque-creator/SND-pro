"""Replace app code from a ZIP without git."""

import zipfile
from pathlib import Path

from sndintel.update import apply_code_zip, find_download_zip


def test_apply_zip_replaces_src_and_keeps_venv(tmp_path):
    src_root = tmp_path / "SND-pro-cursor-actionable-ops-layer-2f34"
    (src_root / "src" / "sndintel").mkdir(parents=True)
    (src_root / "src" / "sndintel" / "marker.txt").write_text("new-code")
    (src_root / "pyproject.toml").write_text("[project]\nname='sndintel'\n")
    archive = tmp_path / "SND-pro-cursor-actionable-ops-layer-2f34.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in src_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(tmp_path))

    dest = tmp_path / "app"
    (dest / ".venv").mkdir(parents=True)
    (dest / ".venv" / "keep-me").write_text("venv")
    (dest / "src" / "sndintel").mkdir(parents=True)
    (dest / "src" / "sndintel" / "marker.txt").write_text("old-code")
    (dest / "warehouse.db").write_text("do-not-copy")

    result = apply_code_zip(archive, dest)
    assert Path(result["app_dir"]) == dest
    assert (dest / "src" / "sndintel" / "marker.txt").read_text() == "new-code"
    assert (dest / ".venv" / "keep-me").read_text() == "venv"
    assert (dest / "warehouse.db").read_text() == "do-not-copy"


def test_find_download_zip_picks_newest(tmp_path):
    older = tmp_path / "SND-pro-old.zip"
    newer = tmp_path / "SND-pro-new.zip"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    older.touch()
    import os
    import time

    os.utime(older, (time.time() - 100, time.time() - 100))
    os.utime(newer, (time.time(), time.time()))
    assert find_download_zip(tmp_path) == newer
