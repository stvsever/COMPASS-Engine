import sys
from pathlib import Path

import tiktoken
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.config.settings import get_settings
from multi_agent_system.utils.core.fusion_layer import FusionLayer


def _feat(fid: str, path):
    return {
        "feature_id": fid,
        "field_name": fid,
        "z_score": -1.2,
        "domain": "BRAIN_MRI",
        "path_in_hierarchy": list(path),
    }


def test_processed_raw_included_when_under_threshold():
    fl = FusionLayer.__new__(FusionLayer)
    fl.settings = get_settings()
    fl.encoder = tiktoken.get_encoding("cl100k_base")
    fl.threshold = 100_000
    fl.embedding_store = SimpleNamespace(db_path=":memory:", fallback_reason=None)

    # No-op backfill for deterministic test.
    def _noop_fill(self, remaining_tokens, candidate_features_by_domain, current_unprocessed, phenotype_text, max_prefilter=2500):
        return current_unprocessed, {"added_count": 0}

    fl._fill_context_with_rag = _noop_fill.__get__(fl, FusionLayer)  # type: ignore

    multimodal = {
        "BRAIN_MRI": [
            _feat("struct_feat", ["structural"]),
            _feat("func_feat", ["functional"]),
        ]
    }

    step_outputs = {
        10: {
            "tool_name": "UnimodalCompressor",
            "domain": "BRAIN_MRI:structural",
            "_step_meta": {
                "step_id": 10,
                "tool_name": "UnimodalCompressor",
                "input_domains": ["BRAIN_MRI"],
                "parameters": {"node_paths": ["BRAIN_MRI|structural"]},
            },
        }
    }

    res = fl.smart_fuse(
        step_outputs=step_outputs,
        hierarchical_deviation={},
        non_numerical_data="notes",
        multimodal_data=multimodal,
        target_condition="neuropsychiatric",
        control_condition="brain-implicated pathology, but NOT psychiatric",
        system_prompt="",
    )

    assert res.raw_processed_multimodal_data is not None
    assert "BRAIN_MRI" in res.raw_processed_multimodal_data


def test_processed_raw_not_included_when_over_threshold():
    fl = FusionLayer.__new__(FusionLayer)
    fl.settings = get_settings()
    fl.encoder = tiktoken.get_encoding("cl100k_base")
    fl.threshold = 1
    fl.embedding_store = SimpleNamespace(db_path=":memory:", fallback_reason=None)

    def _noop_fill(self, remaining_tokens, candidate_features_by_domain, current_unprocessed, phenotype_text, max_prefilter=2500):
        return current_unprocessed, {"added_count": 0}

    fl._fill_context_with_rag = _noop_fill.__get__(fl, FusionLayer)  # type: ignore

    multimodal = {
        "BRAIN_MRI": [
            _feat("struct_feat", ["structural"]),
            _feat("func_feat", ["functional"]),
        ]
    }

    step_outputs = {
        10: {
            "tool_name": "UnimodalCompressor",
            "domain": "BRAIN_MRI:structural",
            "_step_meta": {
                "step_id": 10,
                "tool_name": "UnimodalCompressor",
                "input_domains": ["BRAIN_MRI"],
                "parameters": {"node_paths": ["BRAIN_MRI|structural"]},
            },
        }
    }

    res = fl.smart_fuse(
        step_outputs=step_outputs,
        hierarchical_deviation={},
        non_numerical_data="notes",
        multimodal_data=multimodal,
        target_condition="neuropsychiatric",
        control_condition="brain-implicated pathology, but NOT psychiatric",
        system_prompt="",
    )

    assert res.raw_processed_multimodal_data is None
