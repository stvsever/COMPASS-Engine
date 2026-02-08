import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.utils.core.multimodal_coverage import feature_key_set
from multi_agent_system.utils.core.predictor_input_assembler import PredictorInputAssembler


def _feat(fid: str, path):
    return {
        "feature_id": fid,
        "field_name": fid,
        "domain": "BRAIN_MRI",
        "path_in_hierarchy": list(path),
        "z_score": -1.2,
    }


def test_assembler_uses_locked_priority_order():
    assembler = PredictorInputAssembler(max_chunk_tokens=10000, model_hint="gpt-5")
    executor_output = {
        "non_numerical_data": "non numerical raw",
        "hierarchical_deviation": {"root": {"severity": "MODERATE"}},
        "data_overview": {"coverage": 0.98},
        "step_outputs": {
            1: {"tool_name": "PhenotypeRepresentation", "x": 1},
            2: {"tool_name": "FeatureSynthesizer", "x": 2},
            3: {"tool_name": "DifferentialDiagnosis", "x": 3},
            4: {"tool_name": "ClinicalRelevanceRanker", "x": 4},
            5: {"tool_name": "MultimodalNarrativeCreator", "x": 5},
            6: {"tool_name": "UnimodalCompressor", "x": 6},
            7: {"tool_name": "AnomalyNarrativeBuilder", "x": 7},
            8: {"tool_name": "CodeExecutor", "x": 8},
            9: {"tool_name": "HypothesisGenerator", "x": 9},
            10: {"tool_name": "OtherToolShouldRemain", "x": 10},
        },
    }
    predictor_input = {
        "non_numerical_data_raw": "non numerical raw",
        "hierarchical_deviation_raw": {"root": {"severity": "MODERATE"}},
        "multimodal_unprocessed_raw": {"BRAIN_MRI": {"structural": {"_leaves": [_feat("a", ["structural"])]}}},
        "context_fill_report": {"top_added": [{"domain": "BRAIN_MRI", "feature_name": "a"}]},
    }
    coverage_ledger = {"forced_raw_features": []}

    sections = assembler.build_sections(
        executor_output=executor_output,
        predictor_input=predictor_input,
        coverage_ledger=coverage_ledger,
    )
    names = [s.name for s in sections]

    assert names == [
        "non_numerical_data_raw",
        "hierarchical_deviation_raw",
        "data_overview",
        "phenotype_representation",
        "feature_synthesizer",
        "differential_diagnosis",
        "clinical_relevance_ranker",
        "unprocessed_multimodal_raw",
        "multimodal_narrative",
        "unimodal_outputs",
        "remaining_tool_outputs",
        "auxiliary",
        "processed_multimodal_raw_low_priority",
        "rag_fill",
    ]

    remaining_section = next(s for s in sections if s.name == "remaining_tool_outputs")
    assert "OtherToolShouldRemain" in remaining_section.text


def test_assembler_no_drop_split_reconstructs_text_and_keys():
    big_text = "A" * 12000
    raw_multimodal = {
        "BRAIN_MRI": {
            "structural": {
                "_leaves": [
                    _feat("left_hippo", ["structural", "hippocampus"]),
                    _feat("right_hippo", ["structural", "hippocampus"]),
                ]
            }
        }
    }
    expected_keys = feature_key_set(raw_multimodal)
    assembler = PredictorInputAssembler(max_chunk_tokens=700, model_hint="gpt-5")

    sections = assembler.build_sections(
        executor_output={"step_outputs": {}, "data_overview": {}},
        predictor_input={
            "non_numerical_data_raw": big_text,
            "hierarchical_deviation_raw": {},
            "multimodal_unprocessed_raw": raw_multimodal,
        },
        coverage_ledger={},
    )

    non_num_parts = [s for s in sections if s.name.startswith("non_numerical_data_raw")]
    assert len(non_num_parts) > 1
    assert "".join(s.text for s in non_num_parts) == big_text

    raw_parts = [s for s in sections if s.name.startswith("unprocessed_multimodal_raw")]
    keys_union = set()
    for sec in raw_parts:
        keys_union.update(sec.feature_keys)
    assert keys_union == expected_keys


def test_assembler_chunk_packing_preserves_section_order():
    assembler = PredictorInputAssembler(max_chunk_tokens=900, model_hint="gpt-5")
    sections = assembler.build_sections(
        executor_output={"step_outputs": {}, "data_overview": {"a": 1}},
        predictor_input={
            "non_numerical_data_raw": "n" * 6000,
            "hierarchical_deviation_raw": {"x": 1},
            "multimodal_unprocessed_raw": {},
        },
        coverage_ledger={},
    )
    chunks = assembler.build_chunks(sections)
    flattened_names = [s.name for c in chunks for s in c]
    expected = [s.name for s in sections if s.name != "processed_multimodal_raw_low_priority"]
    assert flattened_names == expected
