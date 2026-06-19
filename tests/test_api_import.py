"""Smoke test: the retrieval API imports and /healthz works without a model/index."""
from __future__ import annotations

import pytest


def test_healthz_without_load():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.serve.api import app

    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["index_loaded"] is False
