"""Shared constants and style configuration for annotated validation."""

from __future__ import annotations

SUPPORTED_PREDICTION_TYPES = {
    "binary",
    "multiclass",
    "regression_univariate",
    "regression_multivariate",
    "hierarchical",
}

PREDICTION_TYPE_ALIASES = {
    "binary_classification": "binary",
    "multiclass_classification": "multiclass",
    "univariate_regression": "regression_univariate",
    "multivariate_regression": "regression_multivariate",
}

ACCEPTED_PREDICTION_TYPES = set(SUPPORTED_PREDICTION_TYPES) | set(PREDICTION_TYPE_ALIASES.keys())


def normalize_prediction_type(value: str) -> str:
    """Normalize user-facing or payload mode names to canonical validation keys."""
    mode = str(value or "binary").strip().lower()
    return PREDICTION_TYPE_ALIASES.get(mode, mode)


PREDICTION_TYPE_LABELS = {
    "binary": "Binary classification",
    "multiclass": "Multi-class classification",
    "regression_univariate": "Univariate regression",
    "regression_multivariate": "Multivariate regression",
    "hierarchical": "Hierarchical mixed task",
}

# Visual style
BG_COLOR = "#0D1117"
CARD_COLOR = "#161B22"
TEXT_PRIMARY = "#E6EDF3"
TEXT_SECONDARY = "#8B949E"
GRID_COLOR = "#30363D"
ACCENT_BLUE = "#58A6FF"
ACCENT_GREEN = "#3FB950"
ACCENT_ORANGE = "#D29922"
ACCENT_RED = "#F85149"
ACCENT_PURPLE = "#BC8CFF"

MODE_PLOT_PREFIX = {
    "binary": "binary",
    "multiclass": "multiclass",
    "regression_univariate": "univariate_regression",
    "regression_multivariate": "multivariate_regression",
    "hierarchical": "hierarchical",
}
