import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.utils.core.plan_executor import PlanExecutor


def _mk_feat(fid: str, path, z=0.0):
    return {
        "feature_id": fid,
        "field_name": fid,
        "z_score": z,
        "domain": "BRAIN_MRI",
        "path_in_hierarchy": list(path),
    }


def test_extract_subtree_from_flat_features_preserves_hierarchy():
    ex = PlanExecutor()
    flat = [
        _mk_feat("left_hippocampus_volume", ["structural", "subcortical_volumes", "hippocampus"], z=-2.1),
        _mk_feat("right_hippocampus_volume", ["structural", "subcortical_volumes", "hippocampus"], z=-1.8),
        _mk_feat("fc_default_mode", ["functional_connectivity", "default_mode"], z=0.4),
    ]

    out = ex._extract_subtrees_robust(
        data=flat,
        domain_name="BRAIN_MRI",
        paths=["BRAIN_MRI|structural|subcortical_volumes"],
    )

    assert isinstance(out, dict)
    assert "structural" in out
    assert "subcortical_volumes" in out["structural"]
    assert "hippocampus" in out["structural"]["subcortical_volumes"]
    leaves = out["structural"]["subcortical_volumes"]["hippocampus"].get("_leaves", [])
    assert len(leaves) == 2
    # Ensure unrelated subtree excluded.
    assert "functional_connectivity" not in out


def test_extract_subtree_unknown_path_failsafe_returns_original():
    ex = PlanExecutor()
    flat = [_mk_feat("a", ["structural", "x"], z=1.0)]

    out = ex._extract_subtrees_robust(
        data=flat,
        domain_name="BRAIN_MRI",
        paths=["BRAIN_MRI|does_not_exist"],
    )

    assert out == flat

