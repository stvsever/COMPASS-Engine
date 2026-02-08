import sys
from pathlib import Path
from types import SimpleNamespace

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.agents.predictor import Predictor
from multi_agent_system.data.models.prediction_result import BinaryClassification


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


def test_call_predictor_json_retries_with_truncated_prompt_on_length_error():
    class _Resp:
        def __init__(self, content):
            self.content = content
            self.prompt_tokens = 10
            self.completion_tokens = 10

    class _Client:
        def __init__(self):
            self.calls = []

        def call(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise ValueError("Model openai/gpt-5-nano returned empty response (finish_reason=length)")
            return _Resp('{"binary_classification":"CONTROL","probability_score":0.4}')

    predictor = Predictor(llm_client=_Client())
    predictor._record_tokens = lambda *_args, **_kwargs: None
    long_prompt = "X " * 12000

    out = predictor._call_predictor_json(
        system_prompt="system",
        user_prompt=long_prompt,
        max_retries=2,
    )

    assert out["binary_classification"] == "CONTROL"
    assert len(predictor.llm_client.calls) >= 2
    second_user = predictor.llm_client.calls[1]["messages"][1]["content"]
    assert "TRUNCATED FOR RETRY" in second_user


def test_predictor_execute_falls_back_on_persistent_llm_failure():
    predictor = Predictor(llm_client=SimpleNamespace())
    predictor._call_predictor_json = lambda **_kwargs: (_ for _ in ()).throw(
        ValueError("Model openai/gpt-5-nano returned empty response (finish_reason=length)")
    )

    result = predictor.execute(
        executor_output={
            "participant_id": "ID3172634",
            "domains_processed": ["BRAIN_MRI"],
            "chunking_skipped": True,
            "predictor_input": {"coverage_ledger": {"all_features": [], "processed_features": []}},
            "non_core_context_text": "compact context",
            "step_outputs": {},
            "data_overview": {},
            "hierarchical_deviation": {},
            "non_numerical_data": "",
            "total_tokens_used": 0,
        },
        target_condition="DEPRESSION",
        control_condition="brain-implicated pathology, but NOT psychiatric",
        iteration=1,
    )

    assert result.binary_classification == BinaryClassification.CONTROL
    assert result.confidence_level.value == "LOW"
    assert any("fallback" in u.lower() for u in result.uncertainty_factors)
