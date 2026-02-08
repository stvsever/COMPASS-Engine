import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.agents.critic import Critic
from multi_agent_system.data.models.prediction_result import Verdict
from multi_agent_system.tools.anomaly_narrative import AnomalyNarrativeBuilder


def test_anomaly_preanalysis_supports_nested_score_schema():
    tool = AnomalyNarrativeBuilder(llm_client=SimpleNamespace())
    deviation = {
        "BIOLOGICAL_ASSAY": {
            "proteomics": {
                "inflammation_markers": {"score": 1.6},
                "neurotrophic_factors": {"score": -2.3},
            }
        },
        "BRAIN_MRI": {
            "structural": {
                "hippocampus": {"score": -2.1},
            }
        },
    }
    analysis = tool._analyze_deviation(deviation)
    assert analysis["total_features"] == 3
    assert analysis["abnormal_features"] >= 2
    assert "BIOLOGICAL_ASSAY" in analysis["domains_summary"]
    assert "BRAIN_MRI" in analysis["domains_summary"]


def test_anomaly_process_output_replaces_false_no_data_claim():
    tool = AnomalyNarrativeBuilder(llm_client=SimpleNamespace())
    deviation = {
        "BRAIN_MRI": {
            "functional_connectivity": {
                "default_mode_network": {"score": 2.8}
            }
        }
    }
    output_data = {
        "integrated_narrative": "No multimodal data available across all domains (total features: 0)."
    }
    repaired = tool._process_output(output_data, {"hierarchical_deviation": deviation})
    assert "No multimodal data available" not in repaired["integrated_narrative"]
    assert repaired.get("integrated_narrative_source") == "deterministic_fallback"
    assert repaired["overall_profile"]["severity"] in {"SEVERE", "MODERATE", "MILD", "NORMAL"}


def test_critic_summary_fallback_uses_reasoning_when_missing():
    critic = Critic(llm_client=SimpleNamespace())
    evaluation_data = {
        "verdict": "SATISFACTORY",
        "composite_score": 0.92,
        "confidence_in_verdict": 0.9,
        "reasoning": "Prediction is coherent and supported by evidence.",
        "checklist": {
            "has_binary_outcome": True,
            "valid_probability": True,
            "sufficient_coverage": True,
            "evidence_based_reasoning": True,
            "clinically_relevant": True,
            "logically_coherent": True,
            "critical_domains_processed": True,
        },
    }
    evaluation = critic._parse_evaluation(evaluation_data, prediction_id="pred_x")
    assert evaluation.verdict == Verdict.SATISFACTORY
    assert evaluation.concise_summary != "No summary provided."
    assert "coherent and supported" in evaluation.concise_summary
