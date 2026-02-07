"""
Participant directory resolution utilities.

Resolves user input (ID, partial ID, or path) to a directory
containing the required COMPASS input files.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Iterable


def _participant_files_match(candidate_dir: Path, settings) -> Tuple[int, int]:
    expected = settings.get_participant_files(candidate_dir)
    present = sum(1 for p in expected.values() if p.exists())
    return present, len(expected)


def _build_id_variants(raw_id: str) -> List[str]:
    raw = str(raw_id or "").strip()
    if not raw:
        return []
    normalized = raw.lower()
    variants = {
        normalized,
        normalized.replace("participant_", ""),
        normalized.replace("participant-", ""),
        normalized.replace("id", "").lstrip("_-"),
        f"id{normalized}",
        f"participant_{normalized}",
        f"participant-id{normalized}",
        f"participant_id{normalized}",
    }
    return [v for v in variants if v]


def _iter_candidate_dirs(root: Path, max_depth: int = 4, max_dirs: int = 2500) -> Iterable[Path]:
    try:
        root = root.resolve()
    except Exception:
        root = root
    if not root.exists() or not root.is_dir():
        return []
    seen = 0
    for dirpath, dirnames, _ in os.walk(root):
        try:
            rel = Path(dirpath).resolve().relative_to(root)
            depth = len(rel.parts)
        except Exception:
            depth = 0
        if depth > max_depth:
            dirnames[:] = []
            continue
        yield Path(dirpath)
        seen += 1
        if seen >= max_dirs:
            break


def _numeric_name_ok(name: str, numeric_id: str) -> bool:
    tokens = re.findall(r"\d+", name)
    if not tokens:
        return False
    return all(token == numeric_id for token in tokens)


def _score_candidate_dir(
    candidate: Path,
    variants: List[str],
    settings,
    numeric_id: Optional[str],
) -> Tuple[int, int, int]:
    present, total = _participant_files_match(candidate, settings)
    name = candidate.name.lower()
    score = 0

    if numeric_id:
        if not _numeric_name_ok(name, numeric_id):
            return 0, present, total

    for variant in variants:
        if name == variant:
            score += 80
        if variant in name:
            score += 35
    if present == total and total > 0:
        score += 120
    score += present * 20
    return score, present, total


def resolve_participant_dir(
    input_id: str,
    compass_data_root: Path,
    settings,
) -> Optional[Path]:
    raw = str(input_id or "").strip()
    if not raw:
        return None

    candidate_paths: List[Path] = []
    raw_path = Path(raw).expanduser()
    candidate_paths.append(raw_path)
    if not raw_path.is_absolute():
        candidate_paths.append(settings.paths.base_dir / raw)
        candidate_paths.append(settings.paths.base_dir.parent / raw)
        candidate_paths.append(compass_data_root / raw)

    for cand in candidate_paths:
        if cand.exists():
            if cand.is_file():
                parent = cand.parent
                present, total = _participant_files_match(parent, settings)
                if present == total:
                    return parent
            elif cand.is_dir():
                present, total = _participant_files_match(cand, settings)
                if present == total:
                    return cand

    numeric_id = raw if raw.isdigit() else None
    variants = _build_id_variants(raw)

    roots: List[Path] = [
        compass_data_root,
        settings.paths.base_dir / "data",
        settings.paths.base_dir.parent,
        settings.paths.base_dir.parent / "data",
    ]

    best: Optional[Path] = None
    best_score = -1
    best_present = -1

    for root in roots:
        for candidate in _iter_candidate_dirs(root, max_depth=4, max_dirs=2500):
            name_l = candidate.name.lower()
            if variants and not any(v in name_l for v in variants):
                continue
            score, present, total = _score_candidate_dir(candidate, variants, settings, numeric_id)
            if score <= 0:
                continue
            if score > best_score or (score == best_score and present > best_present):
                best_score = score
                best_present = present
                if present == total and total > 0:
                    best = candidate
                else:
                    best = candidate

    return best
