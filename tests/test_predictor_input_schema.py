import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.utils.core.fusion_layer import FusionLayer, FusionResult


def test_compress_for_predictor_raw_pass_through_has_canonical_keys():
    fl = FusionLayer.__new__(FusionLayer)
    fr = FusionResult(
        fused_narrative="x",
        domain_summaries={"D": "s"},
        key_findings=[{"k": 1}],
        cross_modal_patterns=[],
        evidence_summary={"for_case": [], "for_control": []},
        tokens_used=0,
        source_outputs=["1"],
        skipped_fusion=True,
        raw_multimodal_data={"BRAIN_MRI": {"structural": {"_leaves": [{"field_name": "a"}]}}},
        raw_step_outputs={1: {"tool_name": "UnimodalCompressor"}},
        context_fill_report={"added_count": 0},
    )

    out = fl.compress_for_predictor(fr, hierarchical_deviation={"root": {}}, non_numerical_data="notes")
    assert out["mode"] == "RAW_PASS_THROUGH"
    assert "domain_summaries" in out
    assert "key_findings" in out
    assert out["multimodal_unprocessed_raw"] == fr.raw_multimodal_data
    assert out["multimodal_context_boost"] == fr.raw_multimodal_data
    assert out["unprocessed_multimodal_data_raw"] == fr.raw_multimodal_data
    assert "multimodal_processed_raw_low_priority" in out
    assert out["context_fill_report"] == fr.context_fill_report


def test_compress_for_predictor_compressed_has_multimodal_aliases():
    fl = FusionLayer.__new__(FusionLayer)
    fr = FusionResult(
        fused_narrative="y",
        domain_summaries={"D": "s"},
        key_findings=[{"k": 1}],
        cross_modal_patterns=[],
        evidence_summary={"for_case": [], "for_control": []},
        tokens_used=10,
        source_outputs=["1"],
        skipped_fusion=False,
        raw_multimodal_data={"BIO": {"_leaves": [{"field_name": "b"}]}},
        raw_step_outputs=None,
        context_fill_report={"added_count": 1},
    )

    out = fl.compress_for_predictor(fr, hierarchical_deviation={"root": {}}, non_numerical_data="notes")
    assert out["mode"] == "COMPRESSED"
    assert out["multimodal_unprocessed_raw"] == fr.raw_multimodal_data
    assert out["multimodal_context_boost"] == fr.raw_multimodal_data
    assert out["unprocessed_multimodal_data_raw"] == fr.raw_multimodal_data
    assert "multimodal_processed_raw_low_priority" in out
