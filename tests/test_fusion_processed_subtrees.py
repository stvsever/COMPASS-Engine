import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tiktoken
from types import SimpleNamespace

from multi_agent_system.config.settings import get_settings
from multi_agent_system.utils.core.fusion_layer import FusionLayer


def test_smart_fuse_excludes_only_processed_subtree_not_whole_domain():
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
            {
                "feature_id": "s1",
                "field_name": "struct_feat",
                "z_score": -2.0,
                "domain": "BRAIN_MRI",
                "path_in_hierarchy": ["structural", "subcortical"],
            },
            {
                "feature_id": "f1",
                "field_name": "func_feat",
                "z_score": 0.5,
                "domain": "BRAIN_MRI",
                "path_in_hierarchy": ["functional"],
            },
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
                "depends_on": [],
            },
        }
    }

    res = fl.smart_fuse(
        step_outputs=step_outputs,
        hierarchical_deviation={},
        non_numerical_data="",
        multimodal_data=multimodal,
        target_condition="neuropsychiatric",
        control_condition="brain-implicated pathology, but NOT psychiatric",
        system_prompt="",
    )

    assert res.skipped_fusion is True
    assert res.raw_multimodal_data is not None
    brain = res.raw_multimodal_data.get("BRAIN_MRI")
    assert isinstance(brain, dict)
    assert "functional" in brain
    assert "structural" not in brain
