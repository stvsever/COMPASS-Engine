import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.data.models.prediction_result import Verdict
import multi_agent_system.main as main_mod
from multi_agent_system.main import (
    _apply_explainability_overrides,
    _parse_xai_methods,
    _run_explainability_for_selected_attempt,
    _select_best_attempt,
)
from multi_agent_system.config.settings import reload_settings


def _evaluation(verdict: Verdict, composite: float, checklist: int, confidence: float = 0.5):
    return SimpleNamespace(
        verdict=verdict,
        composite_score=composite,
        checklist=SimpleNamespace(pass_count=checklist),
        confidence_in_verdict=confidence,
    )


def test_select_best_attempt_prefers_satisfactory():
    attempts = [
        {"iteration": 1, "evaluation": _evaluation(Verdict.UNSATISFACTORY, 0.90, 6)},
        {"iteration": 2, "evaluation": _evaluation(Verdict.SATISFACTORY, 0.55, 5)},
    ]
    selected, reason = _select_best_attempt(attempts)
    assert selected is not None
    assert selected["iteration"] == 2
    assert "Satisfactory verdict available" in reason


def test_select_best_attempt_uses_composite_when_all_unsatisfactory():
    attempts = [
        {"iteration": 1, "evaluation": _evaluation(Verdict.UNSATISFACTORY, 0.40, 4)},
        {"iteration": 2, "evaluation": _evaluation(Verdict.UNSATISFACTORY, 0.70, 4)},
        {"iteration": 3, "evaluation": _evaluation(Verdict.UNSATISFACTORY, 0.70, 5)},
    ]
    selected, reason = _select_best_attempt(attempts)
    assert selected is not None
    assert selected["iteration"] == 3
    assert "No satisfactory verdict" in reason


def test_parse_xai_methods_all_alias():
    assert _parse_xai_methods("all") == ["external", "internal", "hybrid"]


def test_parse_xai_methods_rejects_invalid():
    with pytest.raises(ValueError):
        _parse_xai_methods("external,invalid")


def test_apply_explainability_overrides():
    settings = reload_settings()
    args = SimpleNamespace(
        xai_methods="external,hybrid",
        xai_full_validation=True,
        xai_external_k=9,
        xai_external_runs=2,
        xai_external_adaptive=False,
        xai_internal_model=None,
        xai_internal_steps=None,
        xai_internal_baseline=None,
        xai_internal_span_mode=None,
        xai_hybrid_model="gpt-5-nano",
        xai_hybrid_repeats=3,
        xai_hybrid_temperature=0.2,
    )
    _apply_explainability_overrides(settings, args)
    assert settings.explainability.enabled is True
    assert settings.explainability.methods == ["external", "hybrid"]
    assert settings.explainability.run_full_validation is True
    assert settings.explainability.external_k == 9
    assert settings.explainability.external_runs == 2
    assert settings.explainability.external_adaptive is False
    assert settings.explainability.hybrid_repeats == 3


class _DummyLogger:
    def __init__(self):
        self.payload = None

    def log_explainability(self, payload):
        self.payload = payload


def test_run_explainability_helper_is_fail_safe(monkeypatch, tmp_path):
    settings = reload_settings()
    settings.explainability.enabled = True
    settings.explainability.methods = ["external"]

    def fake_runner(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_mod, "run_explainability_methods", fake_runner)
    logger = _DummyLogger()
    out = _run_explainability_for_selected_attempt(
        settings=settings,
        participant_id="SUBJ_2",
        target_condition="target",
        control_condition="control",
        selected_attempt={"iteration": 1, "executor_output": {}},
        feature_space={"leaf_nodes": ["leaf::a"], "leaf_to_feature": {"leaf::a": {}}, "leaf_to_parent": {}},
        output_dir=tmp_path,
        exec_logger=logger,
    )
    assert out["status"] == "failed"
    assert "Runner failure" in out["reason"]
