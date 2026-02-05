
import sys
import os
from pathlib import Path
from datetime import datetime

import pytest

if not os.getenv("RUN_LLM_TESTS") and not os.getenv("OPENAI_API_KEY"):
    pytest.skip("LLM integration tests disabled (set RUN_LLM_TESTS=1 and OPENAI_API_KEY to enable).", allow_module_level=True)

# Add INFERENCE_PIPELINE root to path (grandparent of this file)
# tests -> multi_agent_system -> INFERENCE_PIPELINE
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multi_agent_system.agents.critic import Critic
from multi_agent_system.data.models.prediction_result import PredictionResult, BinaryClassification, ConfidenceLevel, KeyFinding
from multi_agent_system.data.models.schemas import TargetCondition

def test_critic():
    print("Initializing Critic...")
    critic = Critic()
    
    # Mock data
    prediction = PredictionResult(
        prediction_id="TEST_PRED_001",
        participant_id="TEST_PARTICIPANT",
        target_condition="ANXIETY_DISORDERS",
        created_at=datetime.now(),
        binary_classification=BinaryClassification.CONTROL,
        probability_score=0.2,
        confidence_level=ConfidenceLevel.HIGH,
        key_findings=[
            KeyFinding(domain="BRAIN_MRI", finding="Normal volume", direction="NORMAL", z_score=0.1, relevance_to_prediction="Supports Control"),
        ],
        reasoning_chain=["Data is normal", "No anxiety"],
        supporting_evidence={"for_case": [], "for_control": ["Everything matches control"]},
        uncertainty_factors=[],
        clinical_summary="Patient is normal.",
        domains_processed=["BRAIN_MRI"],
        total_tokens_used=1000,
        iteration=1
    )
    
    executor_output = {
        "execution_result": None,
        "domains_processed": ["BRAIN_MRI"],
        "total_tokens_used": 1000
    }
    
    data_overview = {
        "domain_coverage": {"BRAIN_MRI": {"coverage_percentage": 100, "present_leaves": 10}},
        "target_condition": "ANXIETY_DISORDERS"
    }
    
    hierarchical_deviation = {}
    non_numerical_data = "Notes: Patient is fine."
    
    print("Executing Critic...")
    evaluation = critic.execute(
        prediction=prediction,
        executor_output=executor_output,
        data_overview=data_overview,
        hierarchical_deviation=hierarchical_deviation,
        non_numerical_data=non_numerical_data
    )
    
    print("\n\nRAW EVALUATION OBJECT:")
    print(f"Verdict: {evaluation.verdict}")
    print("\nCHECKLIST PASS COUNT:", evaluation.checklist.pass_count)
    print("CHECKLIST DICT:", evaluation.checklist.dict())
    
    if evaluation.checklist.pass_count > 0:
        print("✅ SUCCESS: Checklist parsed correctly.")
    else:
        print("❌ FAILURE: Checklist is 0/7.")

if __name__ == "__main__":
    test_critic()
