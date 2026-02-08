import sys
from pathlib import Path

import tiktoken
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.agents.communicator import Communicator, _CommSection
from multi_agent_system.config.settings import get_settings


def _build_test_communicator():
    communicator = Communicator.__new__(Communicator)
    communicator.settings = get_settings()
    communicator.LLM_MODEL = "gpt-5-nano"
    communicator.LLM_MAX_TOKENS = 4096
    communicator.LLM_TEMPERATURE = 0.2
    communicator.last_run_metadata = {}
    communicator.embedding_store = SimpleNamespace(db_path=":memory:", fallback_reason=None)
    communicator._log_start = lambda *_args, **_kwargs: None
    communicator._log_complete = lambda *_args, **_kwargs: None
    communicator._build_prompt_header = lambda: "Communicator header"
    communicator._call_llm = lambda _prompt, expect_json=False: {"content": "# Deep Report"}
    try:
        communicator._encoder = tiktoken.encoding_for_model("gpt-5")
    except Exception:
        communicator._encoder = tiktoken.get_encoding("cl100k_base")
    return communicator


def _anchors(focus: str = "", guidance: str = ""):
    return {
        "target_condition": "Target",
        "control_condition": "Control",
        "prediction_output": "{}",
        "critic_evaluation": "{}",
        "execution_summary": "{}",
        "dataflow_summary": "{}",
        "report_context_note": "",
        "user_focus_modalities": focus,
        "user_general_instruction": guidance,
    }


def test_communicator_skips_rag_when_user_guidance_empty():
    communicator = _build_test_communicator()
    communicator._communicator_input_threshold = lambda: 8000
    communicator._build_sections = lambda **_kwargs: (
        [_CommSection("non_numerical_raw", "Small primary evidence.", chunkable=True)],
        _anchors(),
    )

    calls = {"rag": 0}

    def _fake_rag(**_kwargs):
        calls["rag"] += 1
        return [{"path": "x"}]

    communicator._build_guided_rag_rows = _fake_rag

    report = communicator.execute(
        prediction={},
        evaluation={},
        executor_output={},
        data_overview={},
        execution_summary={},
        user_focus_modalities="",
        user_general_instruction="",
    )

    assert report.startswith("# Deep Report")
    assert calls["rag"] == 0
    assert communicator.last_run_metadata["rag_enabled"] is False
    assert communicator.last_run_metadata["rag_added_count"] == 0


def test_communicator_runs_rag_when_user_guidance_present():
    communicator = _build_test_communicator()
    communicator._communicator_input_threshold = lambda: 8000
    communicator._build_sections = lambda **_kwargs: (
        [_CommSection("non_numerical_raw", "Small primary evidence.", chunkable=True)],
        _anchors(focus="Sleep and MRI", guidance="Highlight convergent evidence."),
    )

    calls = {"rag": 0}

    def _fake_rag(**_kwargs):
        calls["rag"] += 1
        return [{"path": "BRAIN_MRI|region|feature", "score": 0.91}]

    communicator._build_guided_rag_rows = _fake_rag

    communicator.execute(
        prediction={},
        evaluation={},
        executor_output={},
        data_overview={},
        execution_summary={},
        user_focus_modalities="Sleep and MRI",
        user_general_instruction="Highlight convergent evidence.",
    )

    assert calls["rag"] == 1
    assert communicator.last_run_metadata["rag_enabled"] is True
    assert communicator.last_run_metadata["rag_added_count"] == 1


def test_communicator_chunk_fallback_only_on_threshold_overflow():
    communicator = _build_test_communicator()
    communicator._communicator_input_threshold = lambda: 20000

    chunk_calls = {"n": 0}

    def _fake_chunk_evidence(**_kwargs):
        chunk_calls["n"] += 1
        return [{"chunk_index": 1, "chunk_total": 1, "summary": "chunked"}]

    communicator._extract_chunk_evidence = _fake_chunk_evidence

    # Fit case: no chunk extraction.
    communicator._build_sections = lambda **_kwargs: (
        [_CommSection("non_numerical_raw", "Short evidence block.", chunkable=True)],
        _anchors(),
    )
    communicator.execute(
        prediction={},
        evaluation={},
        executor_output={},
        data_overview={},
        execution_summary={},
    )
    assert communicator.last_run_metadata["chunking_used"] is False
    assert chunk_calls["n"] == 0

    # Overflow case: chunk extraction required.
    long_text = " ".join(["evidence"] * 90000)
    communicator._build_sections = lambda **_kwargs: (
        [_CommSection("unprocessed_multimodal_raw", long_text, chunkable=True)],
        _anchors(),
    )
    communicator.execute(
        prediction={},
        evaluation={},
        executor_output={},
        data_overview={},
        execution_summary={},
    )
    assert communicator.last_run_metadata["chunking_used"] is True
    assert chunk_calls["n"] == 1
