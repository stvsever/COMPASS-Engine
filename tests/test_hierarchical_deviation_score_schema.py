import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.data.models.execution_plan import PlanStep, ToolName
from multi_agent_system.utils.core.data_loader import DataLoader
from multi_agent_system.utils.core.plan_executor import PlanExecutor


def test_data_loader_parses_nested_score_schema_into_scored_nodes(tmp_path):
    payload = {
        "BIOLOGICAL_ASSAY": {
            "proteomics": {
                "inflammation_markers": {"score": 1.6},
                "neurotrophic_factors": {"score": -2.3},
            },
            "genomics": {"polygenic_risk_scores": {"score": 2.1}},
        },
        "BRAIN_MRI": {
            "structural": {
                "subcortical_volumes": {
                    "hippocampus": {"score": -2.1},
                    "amygdala": {"score": -1.5},
                }
            },
            "functional_connectivity": {
                "resting_state": {"default_mode_network": {"score": 2.8}}
            },
        },
    }
    path = tmp_path / "hierarchical_deviation_map.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loader = DataLoader()
    deviation = loader._load_hierarchical_deviation(path)
    abnormal = deviation.get_abnormal_nodes(min_z=1.5)

    assert len(abnormal) >= 5
    assert any(n.node_name == "default_mode_network" and float(n.z_score or 0.0) == 2.8 for n in abnormal)


def test_plan_executor_anomaly_builder_receives_untruncated_deviation_tree():
    deep = current = {}
    for idx in range(14):
        nxt = {}
        current[f"level_{idx}"] = nxt
        current = nxt
    current["score"] = 2.4

    step = PlanStep(
        step_id=2,
        tool_name=ToolName.ANOMALY_NARRATIVE,
        description="Build narrative",
        reasoning="Test",
        input_domains=["BRAIN_MRI"],
        parameters={},
        expected_output="narrative",
        estimated_tokens=0,
        depends_on=[],
    )
    context = {
        "hierarchical_deviation": {"BRAIN_MRI": deep},
        "data_overview": {},
        "non_numerical_data": "",
        "target_condition": "Major Depressive Disorder",
        "control_condition": "brain-implicated pathology, but NOT psychiatric",
        "participant_id": "SUBJ_TEST",
        "multimodal_data": {},
    }

    pe = PlanExecutor()
    tool_input = pe._build_tool_input(step, context, previous_outputs={})

    node = tool_input["hierarchical_deviation"]["BRAIN_MRI"]
    for idx in range(14):
        node = node[f"level_{idx}"]
    assert float(node["score"]) == 2.4
