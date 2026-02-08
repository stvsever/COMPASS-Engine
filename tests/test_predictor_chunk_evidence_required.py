import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.agents.predictor import Predictor


def test_predictor_requires_chunk_evidence():
    predictor = Predictor.__new__(Predictor)
    predictor._log_start = lambda *_args, **_kwargs: None
    with pytest.raises(ValueError):
        predictor.execute(
            executor_output={},
            target_condition="test-condition",
            control_condition="brain-implicated pathology, but NOT psychiatric",
        )
