import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.frontend import compass_ui
from multi_agent_system.frontend.compass_ui import app


def test_hf_models_endpoint_returns_catalog(monkeypatch):
    def fake_open(req, timeout=15):
        return [
            {"modelId": "Qwen/Qwen2.5-7B", "downloads": 123, "likes": 10, "pipeline_tag": "text-generation"},
            {"modelId": "mistralai/Mistral-7B-Instruct", "downloads": 456, "likes": 20, "pipeline_tag": "text-generation"},
            {"modelId": "sentence-transformers/all-MiniLM-L6-v2", "downloads": 222, "likes": 99, "pipeline_tag": "feature-extraction"},
        ]

    monkeypatch.setattr(compass_ui, "_open_url_json", fake_open)
    client = app.test_client()
    resp = client.get("/api/hf/models")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "models" in body
    assert any(row["id"] == "Qwen/Qwen2.5-7B" for row in body["models"])


def test_hf_models_embedding_task_filters_non_embeddings(monkeypatch):
    def fake_open(req, timeout=15):
        return [
            {"modelId": "Qwen/Qwen2.5-7B", "downloads": 123, "likes": 10, "pipeline_tag": "text-generation"},
            {"modelId": "BAAI/bge-base-en-v1.5", "downloads": 987, "likes": 55, "pipeline_tag": "feature-extraction"},
            {"modelId": "Qwen/Qwen2.5-0.5B-Instruct", "downloads": 400, "likes": 12, "pipeline_tag": "text-generation"},
        ]

    monkeypatch.setattr(compass_ui, "_open_url_json", fake_open)
    client = app.test_client()
    resp = client.get("/api/hf/models?task=embedding")
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["models"]) == 1
    assert body["models"][0]["id"] == "BAAI/bge-base-en-v1.5"
    assert body["models"][0]["is_embedding"] is True


def test_hf_model_detail_endpoint_parses_context(monkeypatch):
    def fake_open(req, timeout=15):
        return {
            "modelId": "Qwen/Qwen2.5-7B",
            "downloads": 999,
            "likes": 55,
            "pipeline_tag": "text-generation",
            "config": {"max_position_embeddings": 4096},
        }

    monkeypatch.setattr(compass_ui, "_open_url_json", fake_open)
    client = app.test_client()
    resp = client.get("/api/hf/model/Qwen%2FQwen2.5-7B")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["id"] == "Qwen/Qwen2.5-7B"
    assert body["context_length"] == 4096
    assert body["is_embedding"] is False


def test_hf_model_detail_endpoint_handles_missing_context(monkeypatch):
    def fake_open(req, timeout=15):
        return {
            "modelId": "sentence-transformers/all-MiniLM-L6-v2",
            "downloads": 500,
            "likes": 100,
            "pipeline_tag": "feature-extraction",
            "config": {},
        }

    monkeypatch.setattr(compass_ui, "_open_url_json", fake_open)
    client = app.test_client()
    resp = client.get("/api/hf/model/sentence-transformers%2Fall-MiniLM-L6-v2")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["context_length"] in (None, 0)
    assert body["is_embedding"] is True


def test_hf_model_detail_prefers_conservative_context_window(monkeypatch):
    def fake_open(req, timeout=15):
        url = getattr(req, "full_url", "")
        if "/api/models/Qwen/Qwen3-Embedding-8B" in url:
            return {
                "modelId": "Qwen/Qwen3-Embedding-8B",
                "downloads": 1000,
                "likes": 100,
                "pipeline_tag": "feature-extraction",
                "config": {"max_position_embeddings": 131072},
            }
        if "/Qwen/Qwen3-Embedding-8B/raw/main/tokenizer_config.json" in url:
            return {"model_max_length": 32768}
        return {}

    monkeypatch.setattr(compass_ui, "_open_url_json", fake_open)
    client = app.test_client()
    resp = client.get("/api/hf/model/Qwen%2FQwen3-Embedding-8B")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["context_length"] == 32768
    assert body["is_embedding"] is True
