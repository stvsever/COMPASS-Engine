import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.frontend.compass_ui import EventStore


def test_step_start_stage_mapping_for_predictor_and_communicator():
    store = EventStore()
    store.add_event("STEP_START", {"id": 910, "tool": "Predictor Agent", "desc": "Predict"})
    assert store.state["current_stage"] == 4

    store.add_event("STEP_START", {"id": 930, "tool": "Communicator Agent", "desc": "Communicate"})
    assert store.state["current_stage"] == 6


def test_explicit_stage_on_step_start_overrides_inference():
    store = EventStore()
    store.add_event("STEP_START", {"id": 4, "tool": "X", "desc": "Y", "stage": 4})
    assert store.state["current_stage"] == 4


def test_deep_report_state_transitions():
    store = EventStore()

    store.add_event("DEEP_REPORT", {"status": "queued", "available": False, "error": None})
    assert store.state["deep_report_status"] == "queued"
    assert store.state["deep_report_available"] is False
    assert store.state["deep_report_error"] is None

    store.add_event("DEEP_REPORT", {"status": "running", "available": False, "error": None})
    assert store.state["deep_report_status"] == "running"

    store.add_event("DEEP_REPORT", {"status": "completed", "available": True, "error": None})
    assert store.state["deep_report_status"] == "completed"
    assert store.state["deep_report_available"] is True
    assert store.state["deep_report_last_generated_at"] is not None

    store.add_event("DEEP_REPORT", {"status": "failed", "available": False, "error": "boom"})
    assert store.state["deep_report_status"] == "failed"
    assert store.state["deep_report_available"] is False
    assert store.state["deep_report_error"] == "boom"
