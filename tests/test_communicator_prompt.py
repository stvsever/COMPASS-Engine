import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.agents.communicator import Communicator
from multi_agent_system.config.settings import get_settings


def test_communicator_prompt_includes_warning_context_note():
    communicator = Communicator.__new__(Communicator)
    communicator.settings = get_settings()

    prompt = communicator._build_prompt(
        prediction={"binary_classification": "CONTROL"},
        evaluation={"verdict": "UNSATISFACTORY"},
        executor_output={"predictor_input": {"context_fill_report": {"added_count": 1}}},
        data_overview={"coverage": 0.97},
        execution_summary={"iterations": 3},
        report_context_note="WARNING: final verdict unsatisfactory",
        user_focus_modalities="Affective neuroimaging and inflammation",
        user_general_instruction="Prioritize contradictory evidence discussion.",
    )

    assert "Final Verdict Context Note" in prompt
    assert "WARNING: final verdict unsatisfactory" in prompt
    assert "Clinical Focus Areas (Optional)" in prompt
    assert "Affective neuroimaging and inflammation" in prompt
    assert "Additional Guidance (Optional)" in prompt
    assert "Prioritize contradictory evidence discussion." in prompt
