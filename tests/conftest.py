from __future__ import annotations

import pytest

from sndintel.sampledata import generate_demo_files


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    return generate_demo_files(out_dir=out, n_shops=60, seed=7, start="2024-01", end="2026-07")
