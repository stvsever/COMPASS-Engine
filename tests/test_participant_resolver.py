from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.config.settings import Settings
from multi_agent_system.utils.participant_resolver import resolve_participant_dir


def _write_required_files(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / "data_overview.json").write_text("{}")
    (base / "hierarchical_deviation_map.json").write_text("{}")
    (base / "multimodal_data.json").write_text("{}")
    (base / "non_numerical_data.txt").write_text("notes")


def test_numeric_id_is_not_fuzzy(tmp_path: Path):
    settings = Settings()
    root = tmp_path / "data_root"
    root.mkdir()
    _write_required_files(root / "participant_001")
    _write_required_files(root / "participant_010")

    resolved = resolve_participant_dir("001", root, settings)
    assert resolved is not None
    assert resolved.name == "participant_001"


def test_path_input_resolves_parent(tmp_path: Path):
    settings = Settings()
    root = tmp_path / "data_root"
    target = root / "SUBJ_001_PSEUDO"
    _write_required_files(target)
    file_path = target / "data_overview.json"

    resolved = resolve_participant_dir(str(file_path), root, settings)
    assert resolved is not None
    assert resolved == target


def test_numeric_id_with_additional_number_tokens_still_matches_exact_id(tmp_path: Path):
    settings = Settings()
    root = tmp_path / "data_root"
    root.mkdir()

    good = root / "participant_3172634_run_2025"
    bad = root / "participant_3172635_run_2025"
    _write_required_files(good)
    _write_required_files(bad)

    resolved = resolve_participant_dir("3172634", root, settings)
    assert resolved is not None
    assert resolved == good


def test_numeric_id_match_can_come_from_parent_path(tmp_path: Path):
    settings = Settings()
    root = tmp_path / "data_root"
    root.mkdir()

    # ID only appears in parent path, not leaf directory name.
    parent = root / "cohort_3172634_batch_01"
    good = parent / "inputs_ready"
    _write_required_files(good)

    # Competing folder with wrong numeric token should not match.
    wrong_parent = root / "cohort_3172635_batch_01"
    wrong = wrong_parent / "inputs_ready"
    _write_required_files(wrong)

    resolved = resolve_participant_dir("3172634", root, settings)
    assert resolved is not None
    assert resolved == good
