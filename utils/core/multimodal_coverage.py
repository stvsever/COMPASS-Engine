"""Helpers for no-loss multimodal coverage accounting and feature-key handling."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple


def canonical_feature_key(domain: str, path: Iterable[Any], feature_name: Any) -> str:
    path_parts = [str(p).strip() for p in (path or []) if str(p).strip()]
    feat = str(feature_name).strip() or "unknown_feature"
    dom = str(domain).strip() or "UNKNOWN_DOMAIN"
    return f"{dom}|{'/'.join(path_parts)}|{feat}"


def _leaf_feature_name(feat: Mapping[str, Any]) -> str:
    return str(
        feat.get("feature_id")
        or feat.get("field_name")
        or feat.get("feature")
        or "unknown_feature"
    )


def flatten_features(data: Any, domain_hint: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Flatten multimodal structures into (feature_key, feature_dict) rows.

    Supports:
    - DataLoader format: {domain: [feature_dict, ...]}
    - Nested tree format with `_leaves` lists
    - Mixed nested dict/list containers
    """

    rows: List[Tuple[str, Dict[str, Any]]] = []

    def walk(obj: Any, current_domain: Optional[str], path_prefix: List[str]) -> None:
        if isinstance(obj, list):
            # Flattened feature list style.
            for item in obj:
                if isinstance(item, Mapping) and (
                    "field_name" in item or "feature" in item or "feature_id" in item
                ):
                    feat = dict(item)
                    feat_domain = str(feat.get("domain") or current_domain or domain_hint or "UNKNOWN_DOMAIN")
                    path = feat.get("path_in_hierarchy") or []
                    if not isinstance(path, list):
                        path = []
                    key = canonical_feature_key(feat_domain, path, _leaf_feature_name(feat))
                    rows.append((key, feat))
                else:
                    walk(item, current_domain, path_prefix)
            return

        if isinstance(obj, Mapping):
            # Nested tree style.
            leaves = obj.get("_leaves")
            if isinstance(leaves, list):
                for leaf in leaves:
                    if not isinstance(leaf, Mapping):
                        continue
                    feat = dict(leaf)
                    feat_domain = str(feat.get("domain") or current_domain or domain_hint or "UNKNOWN_DOMAIN")
                    path = feat.get("path_in_hierarchy")
                    if not isinstance(path, list):
                        path = list(path_prefix)
                    key = canonical_feature_key(feat_domain, path, _leaf_feature_name(feat))
                    rows.append((key, feat))

            for k, v in obj.items():
                if k == "_leaves":
                    continue
                next_domain = current_domain
                next_prefix = path_prefix
                if current_domain is None and isinstance(k, str):
                    next_domain = k
                    next_prefix = []
                elif isinstance(k, str):
                    next_prefix = [*path_prefix, k]
                walk(v, next_domain, next_prefix)
            return

    if isinstance(data, Mapping):
        for top_key, top_val in data.items():
            if isinstance(top_key, str):
                walk(top_val, top_key, [])
            else:
                walk(top_val, domain_hint, [])
    else:
        walk(data, domain_hint, [])

    return rows


def feature_key_set(data: Any, domain_hint: Optional[str] = None) -> Set[str]:
    return {k for k, _ in flatten_features(data, domain_hint=domain_hint)}


def feature_map_by_key(data: Any, domain_hint: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, feat in flatten_features(data, domain_hint=domain_hint):
        out[k] = feat
    return out


def features_to_domain_tree(features: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build nested domain->path tree with `_leaves` from flat feature dicts."""
    out: Dict[str, Any] = {}
    for feat_raw in features:
        if not isinstance(feat_raw, Mapping):
            continue
        feat = dict(feat_raw)
        dom = str(feat.get("domain") or "UNKNOWN_DOMAIN")
        path = feat.get("path_in_hierarchy") or []
        if not isinstance(path, list):
            path = []

        domain_root = out.setdefault(dom, {})
        cur = domain_root
        for seg in path:
            seg_s = str(seg)
            if not seg_s:
                continue
            nxt = cur.get(seg_s)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[seg_s] = nxt
            cur = nxt
        leaves = cur.setdefault("_leaves", [])
        if isinstance(leaves, list):
            leaves.append(feat)
    return out


def features_by_keys(feature_map: Mapping[str, Mapping[str, Any]], keys: Iterable[str]) -> Dict[str, Any]:
    selected: List[Mapping[str, Any]] = []
    for k in keys:
        feat = feature_map.get(k)
        if feat is not None:
            selected.append(feat)
    return features_to_domain_tree(selected)
