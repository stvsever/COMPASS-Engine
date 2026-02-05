import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.utils.path_utils import split_node_path, resolve_requested_subtree


def test_split_node_path_delims():
    assert split_node_path("BRAIN_MRI|structural|fractional_anisotropy") == [
        "BRAIN_MRI",
        "structural",
        "fractional_anisotropy",
    ]
    assert split_node_path("BRAIN_MRI:structural:fractional_anisotropy") == [
        "BRAIN_MRI",
        "structural",
        "fractional_anisotropy",
    ]


def test_resolve_requested_subtree_lexical_underscores():
    flat = [
        {
            "feature_id": "fa_1",
            "field_name": "fractional_anisotropy_left",
            "z_score": 1.2,
            "domain": "BRAIN_MRI",
            "path_in_hierarchy": ["Connectomics", "Structural", "fractional_anisotropy"],
        }
    ]
    # Request uses space instead of underscore; should still resolve lexically.
    resolved = resolve_requested_subtree(
        flat_features=flat,
        domain="BRAIN_MRI",
        requested_segs=["Connectomics", "Structural", "fractional anisotropy"],
        cutoff=0.60,
    )
    assert resolved is not None
    dom, prefix = resolved
    assert dom == "BRAIN_MRI"
    assert prefix == ("Connectomics", "Structural", "fractional_anisotropy")

