from fastapi.testclient import TestClient

from sndintel.api import app
from sndintel.cli import app as cli
from sndintel.ingest.pipeline import run_pipeline
from typer.testing import CliRunner


def test_api_brief(demo, tmp_path, monkeypatch):
    db = tmp_path / "warehouse.db"
    run_pipeline(demo["sales"], shop_path=demo["shops"], db_path=db)
    monkeypatch.setenv("SNDINTEL_DB", str(db))
    # api module binds DB at call time via config.DB_PATH — patch the path used by storage
    import sndintel.config as cfg
    import sndintel.storage as storage

    monkeypatch.setattr(cfg, "DB_PATH", db)
    monkeypatch.setattr(storage, "DB_PATH", db)
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    brief = client.get("/brief")
    assert brief.status_code == 200
    assert isinstance(brief.json(), list)
    assert client.get("/focus").status_code == 200
    water = client.get("/waterfall")
    assert water.status_code == 200
    assert isinstance(water.json(), list)


def test_cli_query_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
