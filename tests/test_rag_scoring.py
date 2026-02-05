import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tiktoken

from multi_agent_system.config.settings import get_settings
from multi_agent_system.utils.core.fusion_layer import FusionLayer


def test_rag_combined_score_uses_abnormality_when_semantic_unavailable():
    fl = FusionLayer.__new__(FusionLayer)
    fl.settings = get_settings()
    fl.encoder = tiktoken.get_encoding("cl100k_base")

    # Force lexical fallback (no embeddings).
    def _raise(*args, **kwargs):
        raise RuntimeError("no embeddings")

    fl._get_cached_embedding = _raise  # type: ignore

    candidates = {
        "BIO": [
            {"feature_id": "f0", "field_name": "f0", "z_score": 0.0, "path_in_hierarchy": ["a"]},
            {"feature_id": "f1", "field_name": "f1", "z_score": 1.5, "path_in_hierarchy": ["a"]},
            {"feature_id": "f2", "field_name": "f2", "z_score": 3.0, "path_in_hierarchy": ["a"]},
        ]
    }

    filled, report = fl._fill_context_with_rag(
        remaining_tokens=10_000,
        candidate_features_by_domain=candidates,
        current_unprocessed={},
        phenotype_text="",  # makes semantic similarity 0 for lexical fallback
    )

    assert report.get("semantic_backend") == "lexical_fallback"
    assert report.get("added_count", 0) >= 1
    top = report.get("top_added", [])
    assert top
    # Highest |z| should be ranked first when semantic contribution is 0.
    assert top[0]["feature_name"] == "f2"
    assert top[0]["abnormality_score"] == 1.0

