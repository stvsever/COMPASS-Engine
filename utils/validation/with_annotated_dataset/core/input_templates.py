"""Template path helpers for annotated validation inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .constants import normalize_prediction_type

TEMPLATE_EXAMPLES_DIR = (
    Path(__file__).resolve().parent.parent / "annotation_templates" / "examples"
)

TEMPLATE_BY_MODE: Dict[str, str] = {
    "binary": "binary_targets_example.json",
    "multiclass": "multiclass_annotations_example.json",
    "regression_univariate": "regression_univariate_annotations_example.json",
    "regression_multivariate": "regression_multivariate_annotations_example.json",
    "hierarchical": "hierarchical_annotations_example.json",
}


def template_file_for_mode(prediction_type: str) -> Path:
    mode = normalize_prediction_type(prediction_type)
    name = TEMPLATE_BY_MODE.get(mode, "binary_targets_example.json")
    return TEMPLATE_EXAMPLES_DIR / name


def template_hint_for_mode(prediction_type: str) -> str:
    return str(template_file_for_mode(prediction_type))
