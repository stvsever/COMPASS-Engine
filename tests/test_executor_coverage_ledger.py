import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.agents.executor import Executor
from multi_agent_system.utils.core.multimodal_coverage import canonical_feature_key


def _feat(fid: str, path):
    return {
        "feature_id": fid,
        "field_name": fid,
        "domain": "BRAIN_MRI",
        "path_in_hierarchy": list(path),
        "z_score": -1.5,
    }


def test_coverage_ledger_forces_missing_features_into_raw_payload():
    executor = Executor.__new__(Executor)

    multimodal_data = {
        "BRAIN_MRI": [
            _feat("left_hippo", ["structural", "hippocampus"]),
            _feat("right_hippo", ["structural", "hippocampus"]),
        ]
    }
    consumed_left = canonical_feature_key(
        "BRAIN_MRI",
        ["structural", "hippocampus"],
        "left_hippo",
    )

    step_outputs = {
        10: {
            "_step_meta": {
                "consumed_feature_keys": [consumed_left],
            }
        }
    }
    predictor_input = {"multimodal_unprocessed_raw": {}}

    ledger = executor._build_coverage_ledger(
        multimodal_data=multimodal_data,
        step_outputs=step_outputs,
        predictor_input=predictor_input,
    )

    assert ledger["summary"]["all_count"] == 2
    assert ledger["summary"]["missing_count"] == 0
    assert ledger["summary"]["forced_raw_count"] == 1
    assert len(ledger["forced_raw_features"]) == 1
    assert predictor_input["multimodal_unprocessed_raw"]
