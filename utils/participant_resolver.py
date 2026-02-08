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
        # Skip noisy/irrelevant trees to stay non-intrusive and fast.
        dirnames[:] = [
            d for d in dirnames
            if d not in {".git", ".idea", "__pycache__", "node_modules", ".venv", "venv"}
        ]
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
    # Strict exact-token numeric matching: allows additional tokens (e.g., run/date),
    # but prevents fuzzy mixups like 001 vs 010.
    return numeric_id in tokens


def _numeric_path_ok(path_text: str, numeric_id: str) -> bool:
    tokens = re.findall(r"\d+", path_text)
    if not tokens:
        return False
    return numeric_id in tokens


def _score_candidate_dir(
    candidate: Path,
    variants: List[str],
    settings,
    numeric_id: Optional[str],
    preferred_root: Optional[Path] = None,
) -> Tuple[int, int, int]:
    present, total = _participant_files_match(candidate, settings)
    name = candidate.name.lower()
    path_l = str(candidate).lower()
    score = 0

    if numeric_id:
        # Enforce exact numeric-token matching across full path.
        if not _numeric_path_ok(path_l, numeric_id):
            return 0, present, total

    for variant in variants:
        if name == variant:
            score += 80
        if variant in name:
            score += 35
        # Path-level match helps when IDs appear in parent folder names.
        if variant in path_l:
            score += 20
    if present == total and total > 0:
        score += 120
    score += present * 20

    if preferred_root is not None:
        try:
            preferred_root = preferred_root.resolve()
            cand_resolved = candidate.resolve()
            if cand_resolved == preferred_root or preferred_root in cand_resolved.parents:
                score += 300
            elif preferred_root.parent == cand_resolved or preferred_root.parent in cand_resolved.parents:
                score += 80
        except Exception:
            pass
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

    primary_roots: List[Path] = [
        compass_data_root,
        compass_data_root.parent,
        compass_data_root.parent.parent,
    ]

    secondary_roots: List[Path] = [
        settings.paths.base_dir,
        settings.paths.base_dir.parent,
    ]

    # Also scan a few ancestors for sibling projects/workspaces.
    try:
        parent = settings.paths.base_dir.parent
        for _ in range(3):
            secondary_roots.append(parent)
            parent = parent.parent
    except Exception:
        pass

    try:
        parent = compass_data_root.parent
        for _ in range(2):
            secondary_roots.append(parent)
            parent = parent.parent
    except Exception:
        pass
    # Non-intrusive fallback roots: bounded scans only if needed.
    home = Path.home()
    fallback_roots: List[Path] = [
        home / "PythonProjects",
        home / "Documents",
        home / "Downloads",
        home,
    ]

    best: Optional[Path] = None
    best_score = -1
    best_present = -1
    best_total = -1

    def _scan_roots(scan_roots: List[Path], max_depth: int, max_dirs: int) -> None:
        nonlocal best, best_score, best_present, best_total
        for root in scan_roots:
            for candidate in _iter_candidate_dirs(root, max_depth=max_depth, max_dirs=max_dirs):
                path_l = str(candidate).lower()
                if numeric_id and not _numeric_path_ok(path_l, numeric_id):
                    continue
                if variants and not any(v in path_l for v in variants):
                    continue
                score, present, total = _score_candidate_dir(
                    candidate,
                    variants,
                    settings,
                    numeric_id,
                    preferred_root=compass_data_root,
                )
                if score <= 0:
                    continue
                if (
                    score > best_score
                    or (score == best_score and present > best_present)
                    or (score == best_score and present == best_present and total > best_total)
                ):
                    best_score = score
                    best_present = present
                    best_total = total
                    best = candidate

    _scan_roots(primary_roots, max_depth=7, max_dirs=12000)
    if best and best_present == best_total and best_total > 0:
        return best

    _scan_roots(secondary_roots, max_depth=6, max_dirs=10000)
    if best and best_present == best_total and best_total > 0:
        return best

    # If still no complete candidate was found, do a bounded broader scan.
    if not best or best_present < best_total:
        _scan_roots(fallback_roots, max_depth=6, max_dirs=15000)

    return best
