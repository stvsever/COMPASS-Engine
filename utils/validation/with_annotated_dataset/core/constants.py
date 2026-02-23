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
BG_COLOR = "#FFFFFF"
CARD_COLOR = "#F6F8FA"
TEXT_PRIMARY = "#24292F"
TEXT_SECONDARY = "#57606A"
GRID_COLOR = "#D0D7DE"
ACCENT_BLUE = "#0969DA"
ACCENT_GREEN = "#1A7F37"
ACCENT_ORANGE = "#BF8700"
ACCENT_RED = "#CF222E"
ACCENT_PURPLE = "#8250DF"

MODE_PLOT_PREFIX = {
    "binary": "binary",
    "multiclass": "multiclass",
    "regression_univariate": "univariate_regression",
    "regression_multivariate": "multivariate_regression",
    "hierarchical": "hierarchical",
}
