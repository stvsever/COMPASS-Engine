import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.agents.integrator import Integrator
from multi_agent_system.config.settings import get_settings
from multi_agent_system.tools.base_tool import ToolOutput, _tool_instances


class _FakeChunkTool:
    def __init__(self):
        self.calls = []

    def execute(self, input_data):
        self.calls.append(input_data)
        return ToolOutput(
            tool_name="ChunkEvidenceExtractor",
            success=True,
            output={
                "summary": "ok",
                "for_case": ["case evidence"],
                "for_control": ["control evidence"],
                "uncertainty_factors": [],
                "key_findings": [],
                "cited_feature_keys": input_data.get("hinted_feature_keys") or [],
            },
        )


def test_integrator_extract_chunk_evidence_uses_tool_and_returns_rows():
    integrator = Integrator.__new__(Integrator)
    integrator.settings = get_settings()
    integrator._chunk_budget_tokens = lambda: 600

    fake_tool = _FakeChunkTool()
    previous = _tool_instances.get("ChunkEvidenceExtractor")
    _tool_instances["ChunkEvidenceExtractor"] = fake_tool

    try:
        result = integrator.extract_chunk_evidence(
            step_outputs={
                1: {
                    "tool_name": "MultimodalNarrativeCreator",
                    "clinical_narrative": "X" * 8000,
                }
            },
            predictor_input={
                "non_numerical_data_raw": "A" * 18000,
                "hierarchical_deviation_raw": {},
                "multimodal_unprocessed_raw": {},
                "context_fill_report": {
                    "predictor_payload_estimate": {
                        "final_tokens": 200000,
                        "threshold": 1000,
                    }
                },
            },
            coverage_ledger={"all_features": [], "processed_features": []},
            data_overview={},
            hierarchical_deviation={},
            non_numerical_data="",
            target_condition="test-condition",
            control_condition="brain-implicated pathology, but NOT psychiatric",
        )
    finally:
        if previous is None:
            _tool_instances.pop("ChunkEvidenceExtractor", None)
        else:
            _tool_instances["ChunkEvidenceExtractor"] = previous

    assert fake_tool.calls
    assert result["predictor_chunk_count"] >= 1
    assert len(result["chunk_evidence"]) == result["predictor_chunk_count"]
    assert "summary" in result["chunk_evidence"][0]


def test_integrator_skips_chunk_evidence_when_payload_fits_threshold():
    integrator = Integrator.__new__(Integrator)
    integrator.settings = get_settings()
    integrator._chunk_budget_tokens = lambda: 600

    fake_tool = _FakeChunkTool()
    previous = _tool_instances.get("ChunkEvidenceExtractor")
    _tool_instances["ChunkEvidenceExtractor"] = fake_tool

    try:
        result = integrator.extract_chunk_evidence(
            step_outputs={
                1: {
                    "tool_name": "MultimodalNarrativeCreator",
                    "clinical_narrative": "X" * 2000,
                }
            },
            predictor_input={
                "non_numerical_data_raw": "A" * 1500,
                "hierarchical_deviation_raw": {},
                "multimodal_unprocessed_raw": {"BRAIN_MRI": {"_leaves": [{"feature": "x"}]}},
                "context_fill_report": {
                    "predictor_payload_estimate": {
                        "final_tokens": 1000,
                        "threshold": 120000,
                    }
                },
            },
            coverage_ledger={"all_features": [], "processed_features": []},
            data_overview={},
            hierarchical_deviation={},
            non_numerical_data="",
            target_condition="test-condition",
            control_condition="brain-implicated pathology, but NOT psychiatric",
        )
    finally:
        if previous is None:
            _tool_instances.pop("ChunkEvidenceExtractor", None)
        else:
            _tool_instances["ChunkEvidenceExtractor"] = previous

    assert fake_tool.calls == []
    assert result["chunking_skipped"] is True
    assert result["predictor_chunk_count"] == 0
    assert result["chunking_reason"] == "payload_fit_excluding_processed_raw"


def test_integrator_processed_raw_does_not_force_chunking():
    integrator = Integrator.__new__(Integrator)
    integrator.settings = get_settings()
    integrator._chunk_budget_tokens = lambda: 300

    fake_tool = _FakeChunkTool()
    previous = _tool_instances.get("ChunkEvidenceExtractor")
    _tool_instances["ChunkEvidenceExtractor"] = fake_tool

    try:
        result = integrator.extract_chunk_evidence(
            step_outputs={
                1: {"tool_name": "MultimodalNarrativeCreator", "clinical_narrative": "short"},
            },
            predictor_input={
                "non_numerical_data_raw": "A" * 1200,
                "hierarchical_deviation_raw": {},
                "multimodal_unprocessed_raw": {"BRAIN_MRI": {"_leaves": [{"feature": "u"}]}},
                "multimodal_processed_raw_low_priority": {"BIOLOGICAL_ASSAY": {"_leaves": [{"feature": "p"}]}},
                "context_fill_report": {
                    "predictor_payload_estimate": {
                        "final_tokens": 150000,
                        "threshold": 100000,
                    }
                },
            },
            coverage_ledger={"all_features": [], "processed_features": []},
            data_overview={},
            hierarchical_deviation={},
            non_numerical_data="",
            target_condition="test-condition",
            control_condition="brain-implicated pathology, but NOT psychiatric",
        )
    finally:
        if previous is None:
            _tool_instances.pop("ChunkEvidenceExtractor", None)
        else:
            _tool_instances["ChunkEvidenceExtractor"] = previous

    assert fake_tool.calls == []
    assert result["chunking_skipped"] is True
    assert result["processed_raw_excluded"] is True


def test_integrator_excludes_processed_raw_from_chunk_candidates():
    integrator = Integrator.__new__(Integrator)
    integrator.settings = get_settings()
    integrator._chunk_budget_tokens = lambda: 800

    fake_tool = _FakeChunkTool()
    previous = _tool_instances.get("ChunkEvidenceExtractor")
    _tool_instances["ChunkEvidenceExtractor"] = fake_tool

    try:
        result = integrator.extract_chunk_evidence(
            step_outputs={
                1: {"tool_name": "MultimodalNarrativeCreator", "clinical_narrative": "X" * 10000},
            },
            predictor_input={
                "non_numerical_data_raw": "A" * 22000,
                "hierarchical_deviation_raw": {},
                "multimodal_unprocessed_raw": {"BRAIN_MRI": {"_leaves": [{"feature": "u"}]}},
                "multimodal_processed_raw_low_priority": {
                    "BIOLOGICAL_ASSAY": {"_leaves": [{"feature": "p"}]}
                },
                "context_fill_report": {
                    "predictor_payload_estimate": {
                        "final_tokens": 240000,
                        "threshold": 500,
                    }
                },
            },
            coverage_ledger={"all_features": [], "processed_features": []},
            data_overview={},
            hierarchical_deviation={},
            non_numerical_data="",
            target_condition="test-condition",
            control_condition="brain-implicated pathology, but NOT psychiatric",
        )
    finally:
        if previous is None:
            _tool_instances.pop("ChunkEvidenceExtractor", None)
        else:
            _tool_instances["ChunkEvidenceExtractor"] = previous

    assert result["chunking_skipped"] is False
    assert result["processed_raw_excluded"] is True
    assert result["predictor_chunk_count"] >= 1
    assert fake_tool.calls
