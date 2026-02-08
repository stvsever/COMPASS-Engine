import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.utils.core.embedding_store import EmbeddingStore


def test_embedding_store_global_and_participant_scopes(tmp_path):
    db_path = tmp_path / "embeddings.sqlite3"
    store = EmbeddingStore(db_path=db_path)

    calls = {"n": 0}

    def fake_embed(text: str, model: str):
        calls["n"] += 1
        base = float(len(text))
        return [base, base + 1.0, base + 2.0]

    v1 = store.get_or_create_global("feature <- domain", "test-model", fake_embed, "feature_path")
    v2 = store.get_or_create_global("feature <- domain", "test-model", fake_embed, "feature_path")
    assert v1 == v2
    assert calls["n"] == 1

    p1 = store.get_or_create_participant("p01", "snippet one", "test-model", fake_embed, "participant_context")
    p2 = store.get_or_create_participant("p01", "snippet one", "test-model", fake_embed, "participant_context")
    p3 = store.get_or_create_participant("p02", "snippet one", "test-model", fake_embed, "participant_context")

    assert p1 == p2
    assert p1 == p3
    # One global + one participant p01 + one participant p02.
    assert calls["n"] == 3

