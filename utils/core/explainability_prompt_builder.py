"""Prompt builders for explainability methods."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple


def _fmt_value(value: float) -> str:
    return f"{float(value):+.2f}"


def _clean_value_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "not_provided"
    text = text.replace("\n", " ").replace("\r", " ").replace("|", "/")
    if len(text) > 220:
        text = text[:220]
    return text


def _coerce_numeric(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def build_prompt_and_spans(
    *,
    target_condition: str,
    control_condition: str,
    leaf_features: Dict[str, Dict],
    active_leaf_ids: Optional[Set[str]] = None,
    predictor_context: str = "",
) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """
    Build template-preserving prompt and character spans for each leaf line.

    If `active_leaf_ids` is provided, inactive leaves are ablated to value=__MISSING__.
    """
    active_set = set(active_leaf_ids) if active_leaf_ids is not None else set(leaf_features.keys())
    target_clean = str(target_condition or "TARGET").strip()
    control_clean = str(control_condition or "CONTROL").strip()

    header_parts = [
        "You are a binary phenotype classifier.",
        f"Output exactly one label: '{target_clean} CASE' or '{target_clean} CONTROL'.",
        f"Control comparator context: {control_clean}.",
        "",
        "Rules:",
        "- Evaluate only the feature lines below.",
        "- value=__MISSING__ and vote=UNKNOWN means ablated/missing.",
        "- Do not output prose.",
        "",
    ]
    if predictor_context:
        header_parts.extend(
            [
                "Predictor evidence context:",
                predictor_context[:12000],
                "",
            ]
        )
    header_parts.append("Feature lines:")
    header = "\n".join(header_parts) + "\n"

    cursor = len(header)
    spans: Dict[str, Tuple[int, int]] = {}
    lines: List[str] = []

    for leaf_id in sorted(leaf_features.keys()):
        leaf = leaf_features[leaf_id]
        domain = str(leaf.get("domain") or "unknown_domain")
        feature_name = str(leaf.get("feature_name") or "unknown_feature")
        path = leaf.get("path_in_hierarchy") or []
        if not isinstance(path, list):
            path = []
        path_str = "/".join(str(p) for p in path if str(p).strip()) or "_root"
        raw_numeric = _coerce_numeric(leaf.get("value_numeric", leaf.get("value", 0.0)))
        raw_text = _clean_value_text(leaf.get("value_text", leaf.get("value", "not_provided")))
        z_score = _coerce_numeric(leaf.get("z_score", 0.0))
        source_type = str(leaf.get("source_type") or "unknown")
        variable_type = str(leaf.get("variable_type") or "unknown")

        if leaf_id in active_set:
            value_text = raw_text
            value_numeric = raw_numeric
            vote = "CASE" if value_numeric > 0 else "CONTROL" if value_numeric < 0 else "NEUTRAL"
        else:
            value_text = "__MISSING__"
            value_numeric = 0.0
            vote = "UNKNOWN"

        line = (
            f"@@ LEAF_ID={leaf_id} | domain={domain} | path={path_str} | "
            f"feature='{feature_name}' | value={value_text} | "
            f"value_numeric={_fmt_value(value_numeric)} | "
            f"z_score={_fmt_value(z_score)} | variable_type={variable_type} | "
            f"source={source_type} | vote={vote} @@\n"
        )
        start = cursor
        end = cursor + len(line)
        spans[leaf_id] = (start, end)
        lines.append(line)
        cursor = end

    footer = f"\nAnswer with one label only.\nLabel: {target_clean}"
    prompt = header + "".join(lines) + footer
    return prompt, spans


def build_prompt_only(
    *,
    target_condition: str,
    control_condition: str,
    leaf_features: Dict[str, Dict],
    active_leaf_ids: Set[str],
    predictor_context: str = "",
) -> str:
    prompt, _ = build_prompt_and_spans(
        target_condition=target_condition,
        control_condition=control_condition,
        leaf_features=leaf_features,
        active_leaf_ids=active_leaf_ids,
        predictor_context=predictor_context,
    )
    return prompt
