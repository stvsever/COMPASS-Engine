import sys
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.agents.predictor import Predictor


def _feat(fid: str, path):
    return {
        "feature_id": fid,
        "field_name": fid,
        "domain": "BRAIN_MRI",
        "path_in_hierarchy": list(path),
    }


def test_validate_feature_representation_uses_chunk_citations_and_raw_paths():
    predictor = Predictor.__new__(Predictor)
    predictor._encoder = tiktoken.get_encoding("cl100k_base")

    key1 = "BRAIN_MRI|structural/hippocampus|left_hippo"
    key2 = "BRAIN_MRI|structural/hippocampus|right_hippo"

    coverage = {
        "all_features": [key1, key2],
        "processed_features": [key1],
    }
    predictor_input = {
        "multimodal_unprocessed_raw": {
            "BRAIN_MRI": {
                "structural": {
                    "hippocampus": {
                        "_leaves": [_feat("right_hippo", ["structural", "hippocampus"])]
                    }
                }
            }
        }
    }
    chunk_evidence = [
        {"cited_feature_keys": [key1]},
        {"cited_feature_keys": []},
    ]

    summary = predictor._validate_feature_representation(
        coverage_ledger=coverage,
        predictor_input=predictor_input,
        chunk_evidence=chunk_evidence,
    )

    assert summary["invariant_ok"] is True
    assert summary["missing_feature_count"] == 0
    assert summary["represented_feature_count"] >= 2


def test_classification_probability_normalization_is_threshold_clamped():
    predictor = Predictor.__new__(Predictor)

    cls, ambiguous = predictor._normalize_classification("CASE", 0.12, "brain-implicated pathology, but NOT psychiatric")
    p = predictor._normalize_probability_for_classification(0.12, cls)
    assert ambiguous is False
    assert p == 0.5

    cls2, ambiguous2 = predictor._normalize_classification("unknown text", 0.91, "brain-implicated pathology, but NOT psychiatric")
    p2 = predictor._normalize_probability_for_classification(0.91, cls2)
    assert ambiguous2 is True
    assert cls2.value.startswith("CONTROL")
    assert p2 < 0.5
