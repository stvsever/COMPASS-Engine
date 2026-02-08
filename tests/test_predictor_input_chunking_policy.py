import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.utils.core.predictor_input_assembler import (
    PredictorInputAssembler,
    PredictorSection,
)


def test_processed_raw_dropped_when_chunking_required():
    assembler = PredictorInputAssembler(max_chunk_tokens=200, model_hint="gpt-5")
    sections = [
        PredictorSection("non_numerical_data_raw", 1, "A" * 500, {}, []),
        PredictorSection("processed_multimodal_raw_low_priority", 12, "B" * 500, {}, []),
    ]
    chunks = assembler.build_chunks(sections)
    flat = [s.name for c in chunks for s in c]
    assert "processed_multimodal_raw_low_priority" not in flat
