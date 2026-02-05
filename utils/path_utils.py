"""
Path Utilities (Lexical-Only)
=============================

COMPASS uses hierarchical paths for selecting multimodal subtrees (e.g.:
`BRAIN_MRI|structural|subcortical_volumes|hippocampus`).

Requirements:
- PURELY lexical matching (no embeddings) to resolve slightly-off LLM paths.
- Support multiple delimiters (`|` and `:` primarily).
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


_PATH_SPLIT_RE = re.compile(r"[|:/>\\]+")
_PATH_EXTRACT_RE = re.compile(r"([A-Za-z0-9_]+(?:[|:][A-Za-z0-9_\- ()]+)+)")


def split_node_path(raw: Union[str, Sequence[str], None]) -> List[str]:
    """
    Split a node path into segments.

    Accepts either:
    - a string path using `|` or `:` delimiters
    - a list/tuple of already-separated segments

    Also supports extracting the first "path-like" substring from a longer
    sentence if the caller passed the entire description by accident.
    """
    if raw is None:
        return []

    if isinstance(raw, (list, tuple)):
        return [str(s).strip() for s in raw if s is not None and str(s).strip()]

    s = str(raw).strip()
    if not s:
        return []

    # Heuristic: if a path is embedded in a longer sentence, extract it.
    m = _PATH_EXTRACT_RE.search(s)
    if m:
        s = m.group(1)

    parts = [p.strip() for p in _PATH_SPLIT_RE.split(s) if p and p.strip()]
    return parts


def normalize_segment(seg: str) -> str:
    """Normalize a single segment for lexical matching."""
    s = str(seg).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]", " ", s)  # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_path(segs: Sequence[str]) -> str:
    """Normalize a sequence of segments into a canonical `|`-joined string."""
    return "|".join(normalize_segment(s) for s in segs if str(s).strip())


def lexical_score(a_norm: str, b_norm: str) -> float:
    """Purely lexical similarity in [0,1]. Inputs should already be normalized."""
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def best_lexical_match(
    target_norm: str,
    candidate_norm_list: Sequence[str],
    cutoff: float = 0.60,
) -> Optional[int]:
    """Return index of best match, or None if nothing meets cutoff."""
    best_i = None
    best_s = -1.0
    for i, cand in enumerate(candidate_norm_list):
        s = lexical_score(target_norm, cand)
        if s > best_s:
            best_s = s
            best_i = i
    if best_i is None or best_s < cutoff:
        return None
    return best_i


def collect_internal_paths(
    flat_features: Sequence[dict],
    domain: str,
) -> Dict[str, Tuple[str, Tuple[str, ...]]]:
    """
    Collect all internal subtree prefixes seen in flattened feature lists.

    Returns a mapping:
      normalized_full_path -> (domain, original_prefix_tuple)

    Where normalized_full_path includes the domain as the first segment.
    """
    internal: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
    dom = str(domain)

    for feat in flat_features:
        if not isinstance(feat, dict):
            continue
        path = feat.get("path_in_hierarchy") or []
        if not isinstance(path, list):
            continue

        # Include domain root (empty prefix) and all prefixes of the path.
        for i in range(0, len(path) + 1):
            prefix = tuple(str(p) for p in path[:i] if str(p).strip())
            norm = normalize_path([dom, *prefix])
            internal.setdefault(norm, (dom, prefix))

    # Ensure at least the root exists even if the list is empty.
    internal.setdefault(normalize_path([dom]), (dom, tuple()))
    return internal


def resolve_requested_subtree(
    flat_features: Sequence[dict],
    domain: str,
    requested_segs: Sequence[str],
    cutoff: float = 0.60,
) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """
    Resolve a requested subtree path (segments) to an internal prefix tuple.

    Purely lexical:
    - normalize segments
    - remove domain prefix if present
    - best match against internal prefixes derived from `path_in_hierarchy`
    """
    dom = str(domain)
    segs = [str(s) for s in requested_segs if s is not None and str(s).strip()]
    if not segs:
        return (dom, tuple())

    # Remove domain prefix if present.
    if normalize_segment(segs[0]) == normalize_segment(dom):
        segs = segs[1:]

    if not segs:
        return (dom, tuple())

    internal = collect_internal_paths(flat_features, dom)
    candidates = list(internal.keys())

    target_norm = normalize_path([dom, *segs])
    idx = best_lexical_match(target_norm, candidates, cutoff=cutoff)
    if idx is None:
        return None
    return internal[candidates[idx]]


def path_is_prefix(
    prefix: Sequence[str],
    full: Sequence[str],
) -> bool:
    """Check prefix relationship with normalization."""
    if len(prefix) > len(full):
        return False
    for a, b in zip(prefix, full):
        if normalize_segment(a) != normalize_segment(b):
            return False
    return True

