import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.config.settings import reload_settings
import multi_agent_system.main as main_mod
from multi_agent_system.main import _apply_explainability_overrides, _parse_xai_methods
import multi_agent_system.utils.core.explainability_runner as runner
from multi_agent_system.utils.core.explainability_feature_space import build_feature_space
from multi_agent_system.utils.core.explainability_prompt_builder import build_prompt_and_spans


def _feature_space_fixture():
    return {
        "root_node": "__XAI_ROOT__",
        "hierarchy_children": {
            "__XAI_ROOT__": ["dom::BRAIN", "dom::BIO"],
            "dom::BRAIN": ["leaf::brain_a"],
            "dom::BIO": ["leaf::bio_a"],
        },
        "leaf_nodes": ["leaf::brain_a", "leaf::bio_a"],
        "parent_nodes": ["dom::BRAIN", "dom::BIO"],
        "leaf_to_parent": {
            "leaf::brain_a": "dom::BRAIN",
            "leaf::bio_a": "dom::BIO",
        },
        "leaf_to_feature": {
            "leaf::brain_a": {
                "domain": "BRAIN",
                "feature_name": "brain_a",
                "path_in_hierarchy": ["MRI"],
                "value": 1.0,
                "value_numeric": 1.0,
                "value_text": "1.0",
                "z_score": 1.2,
                "source_type": "multimodal",
            },
            "leaf::bio_a": {
                "domain": "BIOLOGICAL_ASSAY",
                "feature_name": "bio_a",
                "path_in_hierarchy": ["Serum"],
                "value": -0.4,
                "value_numeric": -0.4,
                "value_text": "-0.4",
                "z_score": -0.8,
                "source_type": "multimodal",
            },
        },
    }


def _selected_attempt_fixture():
    return {
        "iteration": 3,
        "executor_output": {
            "predictor_call_context": {
                "mode": "direct",
                "high_priority_context": "core predictor context",
                "non_core_context": "non core predictor context",
            }
        },
    }


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
        xai_internal_model="Qwen/Qwen2.5-0.5B-Instruct",
        xai_internal_steps=10,
        xai_internal_baseline="mask",
        xai_internal_span_mode="value",
        xai_hybrid_model="gpt-5-nano",
        xai_hybrid_repeats=3,
        xai_hybrid_temperature=0.2,
    )
    _apply_explainability_overrides(settings, args)
    assert settings.explainability.enabled is True
    assert settings.explainability.methods == ["external", "hybrid"]
    assert settings.explainability.external_k == 9
    assert settings.explainability.external_runs == 2
    assert settings.explainability.external_adaptive is False
    assert settings.explainability.internal_steps == 10
    assert settings.explainability.hybrid_repeats == 3


def test_explainability_runner_all_methods(monkeypatch, tmp_path):
    settings = reload_settings()
    settings.explainability.enabled = True
    settings.explainability.methods = ["internal", "external", "hybrid"]
    settings.explainability.hybrid_repeats = 2
    settings.openrouter_api_key = "test-openrouter-key"
    settings.openrouter_base_url = "https://openrouter.ai/api/v1"

    class FakeIG:
        @staticmethod
        def load_model(model_name=None):
            return object(), object(), "cpu"

        @staticmethod
        def prepare_label_tokens(tokenizer, case_str=" CASE", control_str=" CONTROL"):
            return object()

        @staticmethod
        def integrated_gradients_feature_importance(**kwargs):
            return {"leaf::brain_a": 2.0, "leaf::bio_a": 1.0}, {"ok": 1}

    class FakeHFR:
        @staticmethod
        def monte_carlo_hfr_tokenshap(**kwargs):
            return {"leaf::brain_a": 1.0, "leaf::bio_a": -3.0}

    class FakeLLMSelect:
        @staticmethod
        def get_llm_select_scores(**kwargs):
            leaf_runs = [
                {"norm": {"leaf::brain_a": 0.7, "leaf::bio_a": 0.3}},
                {"norm": {"leaf::brain_a": 0.6, "leaf::bio_a": 0.4}},
            ]
            parent_runs = [
                {"norm": {"dom::BRAIN": 0.7, "dom::BIO": 0.3}},
                {"norm": {"dom::BRAIN": 0.6, "dom::BIO": 0.4}},
            ]
            return leaf_runs, parent_runs

    def fake_loader(module_name, filename):
        if filename == "ig_attribution.py":
            return FakeIG
        if filename == "aHFR_TokenSHAP.py":
            return FakeHFR
        if filename == "LLM_select.py":
            return FakeLLMSelect
        raise AssertionError(f"Unexpected module file: {filename}")

    monkeypatch.setattr(runner, "_load_script_module", fake_loader)
    monkeypatch.setattr(runner, "get_llm_client", lambda: object())

    result = runner.run_explainability_methods(
        settings=settings,
        participant_id="SUBJ_X",
        target_condition="major depressive disorder",
        control_condition="control comparator",
        selected_attempt=_selected_attempt_fixture(),
        feature_space=_feature_space_fixture(),
        output_dir=tmp_path,
    )

    assert result["status"] == "success"
    assert result["methods"]["internal"]["status"] == "success"
    assert result["methods"]["external"]["status"] == "success"
    assert result["methods"]["hybrid"]["status"] == "success"
    assert result["methods"]["hybrid"]["uses_participant_values"] is False
    assert sum(result["methods"]["internal"]["leaf_scores_l1"].values()) == pytest.approx(1.0, abs=1e-6)
    assert Path(result["artifact_path"]).exists()


def test_explainability_runner_partial_failure(monkeypatch, tmp_path):
    settings = reload_settings()
    settings.explainability.enabled = True
    settings.explainability.methods = ["internal", "external"]

    class BrokenIG:
        @staticmethod
        def load_model(model_name=None):
            raise RuntimeError("broken iga model")

    class FakeHFR:
        @staticmethod
        def monte_carlo_hfr_tokenshap(**kwargs):
            return {"leaf::brain_a": 0.2, "leaf::bio_a": 0.8}

    def fake_loader(module_name, filename):
        if filename == "ig_attribution.py":
            return BrokenIG
        if filename == "aHFR_TokenSHAP.py":
            return FakeHFR
        raise AssertionError(f"Unexpected module file: {filename}")

    monkeypatch.setattr(runner, "_load_script_module", fake_loader)
    monkeypatch.setattr(runner, "get_llm_client", lambda: object())
    out = runner.run_explainability_methods(
        settings=settings,
        participant_id="SUBJ",
        target_condition="major depressive disorder",
        control_condition="control",
        selected_attempt=_selected_attempt_fixture(),
        feature_space=_feature_space_fixture(),
        output_dir=tmp_path,
    )
    assert out["status"] == "success"
    assert out["methods"]["internal"]["status"] == "failed"
    assert out["methods"]["external"]["status"] == "success"


def test_main_helper_fail_safe(monkeypatch, tmp_path):
    settings = reload_settings()
    settings.explainability.enabled = True
    settings.explainability.methods = ["external"]

    class DummyLogger:
        def __init__(self):
            self.payload = None

        def log_explainability(self, payload):
            self.payload = payload

    def fake_runner(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_mod, "run_explainability_methods", fake_runner)
    logger = DummyLogger()
    out = main_mod._run_explainability_for_selected_attempt(
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


def test_generate_xai_report_skips_without_successful_method(tmp_path):
    class DummyCommunicator:
        def __init__(self):
            self.last_run_metadata = {}

    out = main_mod._generate_xai_explainability_report(
        communicator=DummyCommunicator(),
        xai_result={
            "methods_requested": ["external"],
            "methods": {"external": {"status": "failed", "error": "boom"}},
        },
        prediction={},
        evaluation={},
        execution_summary={},
        target_condition="major depressive disorder",
        control_condition="control comparator",
        base_output_dir=tmp_path,
        participant_id="SUBJ_SKIP",
        trigger_source="cli",
        interactive_ui=False,
    )
    assert out["success"] is False
    assert out["skipped"] is True
    assert "No successful explainability method outputs available" in out["reason"]
    log_path = tmp_path / "execution_log_SUBJ_SKIP.json"
    assert log_path.exists()


def test_generate_xai_report_success(tmp_path):
    class DummyCommunicator:
        def __init__(self):
            self.last_run_metadata = {}

        def execute_xai_report(self, **kwargs):
            self.last_run_metadata = {
                "report_type": "xai",
                "eta_seconds_estimate": 12,
                "duration_seconds": 0.1,
            }
            return "# XAI Report\n\n|A|B|\n|---|---|\n|1|2|"

    out = main_mod._generate_xai_explainability_report(
        communicator=DummyCommunicator(),
        xai_result={
            "methods_requested": ["external", "hybrid"],
            "methods": {
                "external": {"status": "success"},
                "hybrid": {"status": "success"},
            },
        },
        prediction={"binary_classification": "CASE"},
        evaluation={"verdict": "SATISFACTORY"},
        execution_summary={"iterations": 1},
        target_condition="major depressive disorder",
        control_condition="control comparator",
        base_output_dir=tmp_path,
        participant_id="SUBJ_OK",
        trigger_source="cli",
        interactive_ui=False,
    )
    assert out["success"] is True
    assert Path(out["path"]).exists()
    assert out["metadata"]["methods_successful"] == ["external", "hybrid"]
    log_path = tmp_path / "execution_log_SUBJ_OK.json"
    assert log_path.exists()


def test_build_feature_space_includes_multimodal_lexical_and_non_numerical():
    multimodal = {
        "BIOLOGICAL_ASSAY": {
            "proteomics": {
                "_leaves": [
                    {
                        "feature": "C-Reactive Protein (CRP)",
                        "z_score": 1.9,
                        "value": "4.5 mg/L",
                    }
                ]
            }
        }
    }
    non_numerical = (
        "MENTAL STATUS EXAMINATION:\n"
        "Mood: low\n"
        "Affect: restricted\n"
    )
    feature_space = build_feature_space(multimodal, non_numerical_text=non_numerical)

    features = feature_space["features"]
    crp_rows = [row for row in features if row.get("feature_name") == "C-Reactive Protein (CRP)"]
    assert crp_rows, "Expected CRP feature from multimodal data."
    assert crp_rows[0]["value_text"] == "4.5 mg/L"
    assert crp_rows[0]["value_numeric"] == pytest.approx(4.5, abs=1e-6)

    non_num_rows = [
        row for row in features
        if row.get("domain") in {"NON_NUMERICAL_ORDINAL", "NON_NUMERICAL_NOMINAL"}
    ]
    assert non_num_rows, "Expected extracted non-numerical features."
    assert "dom::NON_NUMERICAL_ORDINAL" in feature_space["parent_nodes"]
    assert "dom::NON_NUMERICAL_NOMINAL" in feature_space["parent_nodes"]
    assert "__XAI_ROOT__" in feature_space["hierarchy_children"]

    mood_row = next((row for row in non_num_rows if row.get("feature_name", "").endswith("::Mood")), None)
    assert mood_row is not None
    assert mood_row.get("variable_type") == "ordinal"
    assert mood_row.get("path_in_hierarchy") == []


def test_prompt_builder_keeps_lexical_value_and_ablation():
    leaf_features = {
        "leaf::bio_crp": {
            "domain": "BIOLOGICAL_ASSAY",
            "feature_name": "C-Reactive Protein (CRP)",
            "path_in_hierarchy": ["proteomics", "inflammation_markers"],
            "value": 4.5,
            "value_numeric": 4.5,
            "value_text": "4.5 mg/L",
            "z_score": 1.9,
            "source_type": "multimodal",
        }
    }

    prompt_active, spans_active = build_prompt_and_spans(
        target_condition="major depressive disorder",
        control_condition="control",
        leaf_features=leaf_features,
        active_leaf_ids={"leaf::bio_crp"},
        predictor_context="",
    )
    assert "value=4.5 mg/L" in prompt_active
    s, e = spans_active["leaf::bio_crp"]
    assert "value=4.5 mg/L" in prompt_active[s:e]

    prompt_ablated, _ = build_prompt_and_spans(
        target_condition="major depressive disorder",
        control_condition="control",
        leaf_features=leaf_features,
        active_leaf_ids=set(),
        predictor_context="",
    )
    assert "value=__MISSING__" in prompt_ablated
