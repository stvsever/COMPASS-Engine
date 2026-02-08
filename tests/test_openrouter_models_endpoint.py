import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.config.settings import get_settings, reload_settings
from multi_agent_system.frontend.compass_ui import app


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_openrouter_models_requires_api_key():
    settings = reload_settings()
    settings.openrouter_api_key = ""
    client = app.test_client()
    resp = client.get("/api/openrouter/models")
    assert resp.status_code == 400


def test_openrouter_models_endpoint_returns_catalog(monkeypatch):
    settings = get_settings()
    settings.openrouter_api_key = "test-key"

    import multi_agent_system.frontend.compass_ui as compass_ui
    monkeypatch.setattr(
        compass_ui,
        "urlopen",
        lambda req, timeout=15, context=None: _FakeResponse(
            {
                "data": [
                    {
                        "id": "openai/gpt-5-nano",
                        "context_length": 128000,
                        "pricing": {"prompt": "0.1", "completion": "0.4"},
                        "architecture": {"modality": "text->text"},
                    },
                    {"id": "openai/gpt-4o-mini", "context_length": 128000, "pricing": {"prompt": "0.05", "completion": "0.15"}},
                    {
                        "id": "openai/text-embedding-3-large",
                        "context_length": 8192,
                        "pricing": {"prompt": "0.02", "completion": "0"},
                        "architecture": {"modality": "text->text"},
                        "supported_parameters": ["input", "embedding"],
                    },
                ]
            }
        ),
    )
    client = app.test_client()
    resp = client.get("/api/openrouter/models")
    body = resp.get_json()
    assert resp.status_code == 200
    assert "models" in body
    assert any(row["id"] == "openai/gpt-5-nano" for row in body["models"])
    assert any(row["id"] == "openai/text-embedding-3-large" and row["is_embedding"] for row in body["models"])


def test_openrouter_embedding_models_endpoint_returns_catalog(monkeypatch):
    settings = get_settings()
    settings.openrouter_api_key = "test-key"

    import multi_agent_system.frontend.compass_ui as compass_ui

    def _fake_urlopen(req, timeout=15, context=None):
        url = getattr(req, "full_url", "")
        if url.endswith("/embeddings/models"):
            return _FakeResponse(
                {
                    "data": [
                        {"id": "openai/text-embedding-3-large", "context_length": 8192, "pricing": {"prompt": "0.02"}},
                        {"id": "qwen/qwen3-embedding-8b", "context_length": 32768, "pricing": {"prompt": "0.03"}},
                    ]
                }
            )
        return _FakeResponse({"data": []})

    monkeypatch.setattr(compass_ui, "urlopen", _fake_urlopen)
    client = app.test_client()
    resp = client.get("/api/openrouter/embedding-models")
    body = resp.get_json()
    assert resp.status_code == 200
    assert "models" in body
    assert len(body["models"]) == 2
    assert all(row["is_embedding"] for row in body["models"])
    assert any(row["id"] == "qwen/qwen3-embedding-8b" and row["context_length"] == 32768 for row in body["models"])
