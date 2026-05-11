import asyncio
from types import SimpleNamespace

from app.services.health import build_health_report
from app.services import health


class DummyDB:
    async def execute(self, statement):
        return SimpleNamespace(statement=statement)


class FailingDB:
    async def execute(self, statement):
        raise RuntimeError("database is unavailable")


def test_build_health_report_marks_ok_when_database_and_model_are_ready(monkeypatch):
    monkeypatch.setattr(
        health.PlagiarismService,
        "_instance",
        SimpleNamespace(_initialized=True, device="cpu", hidden_size=768),
        raising=False,
    )

    report = asyncio.run(build_health_report(DummyDB()))

    assert report["status"] == "ok"
    assert report["db"]["status"] == "ok"
    assert report["model"]["status"] == "ok"
    assert report["model"]["ready"] is True


def test_build_health_report_marks_error_when_database_fails(monkeypatch):
    monkeypatch.setattr(health.PlagiarismService, "_instance", None, raising=False)

    report = asyncio.run(build_health_report(FailingDB()))

    assert report["status"] == "error"
    assert report["db"]["status"] == "error"
    assert report["model"]["status"] == "degraded"
