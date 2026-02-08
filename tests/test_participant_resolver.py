from pathlib import Path

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
