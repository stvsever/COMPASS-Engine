"""Utilities to build a reusable feature space for explainability methods."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)")
_SECTION_PREFIX_RE = re.compile(r"^[A-Z][A-Z0-9 _\-/()]{2,}$")
_MAX_NON_NUMERIC_FEATURES = 180
_MAX_NON_NUMERIC_PER_SECTION = 16
_NON_NUM_ORDINAL_TERMS = {
    "none",
    "mild",
    "moderate",
    "severe",
    "very severe",
    "low",
    "medium",
    "high",
    "very high",
    "normal",
    "abnormal",
    "elevated",
    "reduced",
    "increased",
    "decreased",
    "borderline",
    "positive",
    "negative",
    "poor",
    "fair",
    "good",
    "excellent",
    "rarely",
    "sometimes",
    "often",
    "always",
}
_NON_NUM_NOMINAL_TERMS = {
    "male",
    "female",
    "other",
    "yes",
    "no",
    "true",
    "false",
    "present",
    "absent",
}
_MEASUREMENT_HINT_RE = re.compile(
    r"(\d+\s*/\s*\d+|\b\d+(?:\.\d+)?\s*(mg|kg|g|mm|cm|ml|mmol|bpm|hrs?|hours?|years?|mm3|%)\b|[<>]=?\s*\d)",
    re.IGNORECASE,
)


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _extract_first_float(text: str) -> float:
    if not text:
        return 0.0
    match = _FLOAT_RE.search(str(text))
    if not match:
        return 0.0
    try:
        return float(match.group(0))
    except Exception:
        return 0.0


def _coerce_value_text(value: Any) -> str:
    if value is None:
        return "not_provided"
    text = str(value).strip()
    return text if text else "not_provided"


def _value_text_and_numeric(row: Dict[str, Any]) -> Tuple[str, float]:
    value_candidate = None
    for key in ("value", "raw_value", "display_value", "feature_value", "field_value"):
        if key in row and row.get(key) not in (None, ""):
            value_candidate = row.get(key)
            break
    z_score = _safe_float(row.get("z_score", 0.0))

    if value_candidate is None:
        if "z_score" in row:
            value_text = f"{z_score:+.4f}"
            return value_text, z_score
        return "not_provided", 0.0

    value_text = _coerce_value_text(value_candidate)
    numeric = _safe_float(value_candidate)
    if numeric == 0.0 and value_text not in ("0", "0.0", "+0", "-0"):
        numeric = _extract_first_float(value_text)
    if numeric == 0.0 and z_score != 0.0:
        numeric = z_score
    return value_text, numeric


def _leaf_key(domain: str, path: List[str], feature_name: str, source_type: str) -> str:
    path_str = "/".join(path) if path else "_root"
    return f"{source_type}|{domain}|{path_str}|{feature_name}"


@dataclass
class FeatureLeaf:
    leaf_id: str
    domain: str
    parent_id: str
    path_in_hierarchy: List[str]
    feature_name: str
    value: float
    value_numeric: float
    value_text: str
    z_score: float
    variable_type: str
    source_type: str
    raw: Dict[str, Any]


def _iter_domain_features(domain: str, payload: Any, current_path: List[str]) -> List[Tuple[Dict[str, Any], List[str]]]:
    rows: List[Tuple[Dict[str, Any], List[str]]] = []
    if isinstance(payload, list):
        for item in payload:
            item_dict = _to_dict(item)
            if not item_dict:
                continue
            if item_dict.get("feature") or item_dict.get("field_name") or item_dict.get("feature_id"):
                explicit_path = item_dict.get("path_in_hierarchy")
                if isinstance(explicit_path, list):
                    row_path = [str(p) for p in explicit_path if str(p).strip()]
                else:
                    row_path = list(current_path)
                rows.append((item_dict, row_path))
        return rows

    if isinstance(payload, dict):
        leaves = payload.get("_leaves")
        if isinstance(leaves, list):
            for item in leaves:
                item_dict = _to_dict(item)
                if not item_dict:
                    continue
                explicit_path = item_dict.get("path_in_hierarchy")
                if isinstance(explicit_path, list):
                    row_path = [str(p) for p in explicit_path if str(p).strip()]
                else:
                    row_path = list(current_path)
                rows.append((item_dict, row_path))
        for key, value in payload.items():
            if key == "_leaves":
                continue
            if isinstance(value, (dict, list)):
                rows.extend(_iter_domain_features(domain, value, current_path + [str(key)]))
        return rows

    return rows


def _looks_like_header(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    if text.startswith("##"):
        return True
    if text.endswith(":"):
        body = text[:-1].strip()
        return bool(body) and _SECTION_PREFIX_RE.match(body) is not None
    return False


def _parse_non_numerical_sections(raw_text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current = "GENERAL"
    sections[current] = []
    for raw_line in str(raw_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _looks_like_header(line):
            if line.startswith("##"):
                current = line.lstrip("#").strip().strip(":") or "GENERAL"
            else:
                current = line[:-1].strip() or "GENERAL"
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _iter_non_numerical_features(
    raw_text: str,
) -> List[Tuple[str, str, str, float, str, Dict[str, Any]]]:
    out: List[Tuple[str, str, str, float, str, Dict[str, Any]]] = []
    sections = _parse_non_numerical_sections(raw_text)

    def _infer_non_numerical_variable_type(value_text: str, value_numeric: float) -> str:
        value_l = value_text.lower().strip()
        has_digit = any(ch.isdigit() for ch in value_text)
        has_alpha = any(ch.isalpha() for ch in value_text)

        if value_l in _NON_NUM_ORDINAL_TERMS:
            return "ordinal"
        if value_l in _NON_NUM_NOMINAL_TERMS:
            return "nominal"
        if has_digit and has_alpha:
            if _MEASUREMENT_HINT_RE.search(value_text):
                return "ordinal"
            return "nominal"
        if has_digit:
            return "ordinal"
        if value_numeric != 0.0:
            return "ordinal"
        return "nominal"

    for section, lines in sections.items():
        if not lines:
            continue
        section_count = 0
        for line in lines:
            candidate = line.lstrip("-* ").strip()
            if ":" not in candidate:
                continue
            left, right = candidate.split(":", 1)
            feature_name = left.strip()
            value_text = right.strip()
            if not feature_name or not value_text:
                continue
            if len(feature_name) > 80:
                continue
            value_text = value_text[:220]
            value_numeric = _extract_first_float(value_text)
            variable_type = _infer_non_numerical_variable_type(value_text, value_numeric)
            out.append(
                (
                    section,
                    feature_name,
                    value_text,
                    value_numeric,
                    variable_type,
                    {
                        "source_line": line,
                        "section": section,
                        "kind": "kv_pair",
                    },
                )
            )
            section_count += 1
            if section_count >= _MAX_NON_NUMERIC_PER_SECTION:
                break

    if len(out) > _MAX_NON_NUMERIC_FEATURES:
        return out[:_MAX_NON_NUMERIC_FEATURES]
    return out


def build_feature_space(
    multimodal_data: Dict[str, Any],
    *,
    non_numerical_text: str = "",
) -> Dict[str, Any]:
    """
    Build a reusable feature catalog + hierarchy from multimodal and non-numerical data.

    Returns keys:
      - root_node
      - hierarchy_children
      - leaf_nodes
      - parent_nodes
      - leaf_to_parent
      - leaf_to_feature
      - features
    """
    root_node = "__XAI_ROOT__"
    hierarchy_children: Dict[str, List[str]] = {root_node: []}
    leaf_to_parent: Dict[str, str] = {}
    leaf_to_feature: Dict[str, Dict[str, Any]] = {}
    parent_nodes: List[str] = []
    features: List[Dict[str, Any]] = []

    def ensure_edge(parent: str, child: str) -> None:
        children = hierarchy_children.setdefault(parent, [])
        if child not in children:
            children.append(child)

    def register_leaf(
        *,
        source_type: str,
        domain_name: str,
        row: Dict[str, Any],
        path: List[str],
        parent_node: str,
        seen_leaf_ids: set[str],
        variable_type: str = "continuous",
    ) -> None:
        feature_name = str(
            row.get("feature")
            or row.get("field_name")
            or row.get("feature_id")
            or row.get("name")
            or "unknown_feature"
        )
        z_score = _safe_float(row.get("z_score", 0.0))
        value_text, value_numeric = _value_text_and_numeric(row)
        base_leaf = _leaf_key(domain_name, path, feature_name, source_type=source_type)
        leaf_id = f"leaf::{base_leaf}"
        if leaf_id in seen_leaf_ids:
            suffix = 2
            while f"{leaf_id}__{suffix}" in seen_leaf_ids:
                suffix += 1
            leaf_id = f"{leaf_id}__{suffix}"
        seen_leaf_ids.add(leaf_id)

        current_parent = parent_node
        running: List[str] = []
        for seg in path:
            seg_str = str(seg).strip()
            if not seg_str:
                continue
            running.append(seg_str)
            path_node = f"path::{domain_name}::{'/'.join(running)}"
            ensure_edge(current_parent, path_node)
            current_parent = path_node

        ensure_edge(current_parent, leaf_id)

        leaf = FeatureLeaf(
            leaf_id=leaf_id,
            domain=domain_name,
            parent_id=parent_node,
            path_in_hierarchy=path,
            feature_name=feature_name,
            value=value_numeric,
            value_numeric=value_numeric,
            value_text=value_text,
            z_score=z_score,
            variable_type=variable_type,
            source_type=source_type,
            raw=row,
        )
        leaf_dict = asdict(leaf)
        leaf_to_parent[leaf_id] = parent_node
        leaf_to_feature[leaf_id] = leaf_dict
        features.append(leaf_dict)

    multimodal_payload = multimodal_data or {}
    for domain, domain_payload in multimodal_payload.items():
        domain_name = str(domain).strip() or "UNKNOWN_DOMAIN"
        parent_node = f"dom::{domain_name}"
        parent_nodes.append(parent_node)
        ensure_edge(root_node, parent_node)

        seen_leaf_ids: set[str] = set()
        for row, path in _iter_domain_features(domain_name, domain_payload, []):
            register_leaf(
                source_type="multimodal",
                domain_name=domain_name,
                row=row,
                path=path,
                parent_node=parent_node,
                seen_leaf_ids=seen_leaf_ids,
                variable_type="continuous",
            )

    non_num_entries = _iter_non_numerical_features(non_numerical_text)
    if non_num_entries:
        parent_ordinal = "dom::NON_NUMERICAL_ORDINAL"
        parent_nominal = "dom::NON_NUMERICAL_NOMINAL"
        parent_nodes.extend([parent_ordinal, parent_nominal])
        ensure_edge(root_node, parent_ordinal)
        ensure_edge(root_node, parent_nominal)
        seen_leaf_ids: set[str] = set()
        for section, feature_name, value_text, value_numeric, variable_type, raw in non_num_entries:
            domain_name = (
                "NON_NUMERICAL_ORDINAL"
                if variable_type == "ordinal"
                else "NON_NUMERICAL_NOMINAL"
            )
            parent_node = parent_ordinal if variable_type == "ordinal" else parent_nominal
            row = {
                "feature": f"{section}::{feature_name}",
                "value": value_text,
                "z_score": value_numeric,
                **raw,
            }
            register_leaf(
                source_type="non_numerical",
                domain_name=domain_name,
                row=row,
                path=[],
                parent_node=parent_node,
                seen_leaf_ids=seen_leaf_ids,
                variable_type=variable_type,
            )

    for parent, children in list(hierarchy_children.items()):
        hierarchy_children[parent] = sorted(children)

    leaf_nodes = sorted(leaf_to_feature.keys())
    parent_nodes = sorted(set(parent_nodes))

    return {
        "root_node": root_node,
        "hierarchy_children": hierarchy_children,
        "leaf_nodes": leaf_nodes,
        "parent_nodes": parent_nodes,
        "leaf_to_parent": leaf_to_parent,
        "leaf_to_feature": leaf_to_feature,
        "features": features,
    }


def normalize_abs(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(abs(float(v)) for v in scores.values())
    if total <= 0:
        return {k: 0.0 for k in scores}
    return {k: abs(float(v)) / total for k, v in scores.items()}


def aggregate_leaf_scores_to_parent(
    leaf_scores: Dict[str, float],
    leaf_to_parent: Dict[str, str],
) -> Dict[str, float]:
    parent_scores: Dict[str, float] = {}
    for leaf_id, value in (leaf_scores or {}).items():
        parent_id = leaf_to_parent.get(leaf_id)
        if not parent_id:
            continue
        parent_scores[parent_id] = parent_scores.get(parent_id, 0.0) + abs(float(value))
    return parent_scores
