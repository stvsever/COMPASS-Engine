"""
COMPASS Prediction Result Models

Defines the structure for prediction outputs and critic evaluations.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class BinaryClassification(str, Enum):
    """Binary prediction outcome."""
    CASE = "CASE"
    CONTROL = "CONTROL"


class ConfidenceLevel(str, Enum):
    """Confidence in prediction."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Verdict(str, Enum):
    """Critic verdict on prediction quality."""
    SATISFACTORY = "SATISFACTORY"
    UNSATISFACTORY = "UNSATISFACTORY"


class ImprovementPriority(str, Enum):
    """Priority level for improvement suggestions."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ============================================================================
# Prediction Result
# ============================================================================

class KeyFinding(BaseModel):
    """A key finding contributing to the prediction."""
    domain: str
    finding: str
    direction: str = Field(..., description="ABNORMAL_HIGH, ABNORMAL_LOW, or NORMAL")
    z_score: Optional[float] = None
    relevance_to_prediction: str


class PredictionResult(BaseModel):
    """
    Final prediction result from the Predictor agent.
    
    Contains the binary classification, probability score, and supporting evidence.
    """
    prediction_id: str = Field(..., description="Unique prediction identifier")
    participant_id: str
    target_condition: str = Field(..., description="neuropsychiatric or neurologic")
    control_condition: str = Field(..., description="control comparator string")
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Core prediction
    binary_classification: BinaryClassification
    probability_score: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    
    # Evidence
    key_findings: List[KeyFinding] = Field(default_factory=list)
    reasoning_chain: List[str] = Field(default_factory=list)
    supporting_evidence: Dict[str, List[str]] = Field(
        default_factory=lambda: {"for_case": [], "for_control": []}
    )
    uncertainty_factors: List[str] = Field(default_factory=list)
    
    # Summary
    clinical_summary: str = Field("", description="One paragraph clinical summary")
    
    # Execution context
    domains_processed: List[str] = Field(default_factory=list)
    total_tokens_used: int = 0
    iteration: int = 1
    
    @validator('probability_score')
    def validate_probability(cls, v, values):
        """Ensure probability aligns with classification."""
        if 'binary_classification' in values:
            classification = values['binary_classification']
            if classification == BinaryClassification.CASE and v < 0.5:
                raise ValueError(
                    f"Probability {v} should be >= 0.5 for CASE classification"
                )
            if classification == BinaryClassification.CONTROL and v >= 0.5:
                raise ValueError(
                    f"Probability {v} should be < 0.5 for CONTROL classification"
                )
        return v
    
    def to_report_dict(self) -> Dict[str, Any]:
        """Generate dictionary suitable for patient report."""
        return {
            "participant_id": self.participant_id,
            "condition": self.target_condition,
            "control_condition": self.control_condition,
            "prediction": self.binary_classification.value,
            "probability": f"{self.probability_score:.2%}",
            "confidence": self.confidence_level.value,
            "key_findings": [f.finding for f in self.key_findings[:5]],
            "summary": self.clinical_summary
        }


# ============================================================================
# Critic Evaluation
# ============================================================================

class EvaluationChecklist(BaseModel):
    """Checklist of quality criteria."""
    has_binary_outcome: bool = False
    valid_probability: bool = False
    sufficient_coverage: bool = False
    evidence_based_reasoning: bool = False
    clinically_relevant: bool = False
    logically_coherent: bool = False
    critical_domains_processed: bool = False
    
    @property
    def all_passed(self) -> bool:
        """Check if all criteria passed."""
        return all([
            self.has_binary_outcome,
            self.valid_probability,
            self.sufficient_coverage,
            self.evidence_based_reasoning,
            self.clinically_relevant,
            self.logically_coherent,
            self.critical_domains_processed
        ])
    
    @property
    def pass_count(self) -> int:
        """Count of passed criteria."""
        return sum([
            self.has_binary_outcome,
            self.valid_probability,
            self.sufficient_coverage,
            self.evidence_based_reasoning,
            self.clinically_relevant,
            self.logically_coherent,
            self.critical_domains_processed
        ])


class ImprovementSuggestion(BaseModel):
    """A suggestion for improving the prediction."""
    issue: str
    suggestion: str
    priority: ImprovementPriority


class CriticEvaluation(BaseModel):
    """
    Evaluation result from the Critic agent.
    
    Determines whether prediction meets quality standards and provides
    feedback for re-orchestration if needed.
    """
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    prediction_id: str = Field(..., description="ID of evaluated prediction")
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Verdict
    verdict: Verdict
    confidence_in_verdict: float = Field(..., ge=0.0, le=1.0)
    
    # Weighted Scoring
    composite_score: float = Field(..., ge=0.0, le=1.0, description="Weighted score (0-1)")
    score_breakdown: Dict[str, float] = Field(default_factory=dict, description="Breakdown of component scores")
    
    # Detailed assessment
    checklist: EvaluationChecklist
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    
    # Improvement guidance (if unsatisfactory)
    improvement_suggestions: List[ImprovementSuggestion] = Field(default_factory=list)
    domains_missed: List[str] = Field(default_factory=list)
    
    # Reasoning
    reasoning: str = Field("", description="Detailed explanation of evaluation")
    concise_summary: str = Field("", description="Concise summary (1-2 sentences) of why verdict was reached")
    
    @property
    def is_satisfactory(self) -> bool:
        """Quick check for satisfactory verdict."""
        return self.verdict == Verdict.SATISFACTORY
    
    @property
    def high_priority_issues(self) -> List[ImprovementSuggestion]:
        """Get high priority improvement suggestions."""
        return [s for s in self.improvement_suggestions if s.priority == ImprovementPriority.HIGH]
    
    def get_feedback_for_reorchestration(self) -> str:
        """Generate feedback string for the orchestrator."""
        if self.is_satisfactory:
            return ""
        
        feedback_parts = ["PREVIOUS ATTEMPT FEEDBACK:"]
        
        if self.weaknesses:
            feedback_parts.append(f"Weaknesses: {'; '.join(self.weaknesses)}")
        
        if self.domains_missed:
            feedback_parts.append(f"Domains not processed: {', '.join(self.domains_missed)}")
        
        for suggestion in self.high_priority_issues:
            feedback_parts.append(f"HIGH PRIORITY: {suggestion.issue} - {suggestion.suggestion}")
        
        return "\n".join(feedback_parts)


# ============================================================================
# Pipeline Result
# ============================================================================

class PipelineResult(BaseModel):
    """
    Complete result from the COMPASS pipeline for one participant.
    """
    participant_id: str
    target_condition: str
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Final outputs
    final_prediction: PredictionResult
    final_evaluation: CriticEvaluation
    
    # Execution history
    total_iterations: int = 1
    total_tokens_used: int = 0
    total_execution_time_ms: int = 0
    
    # Logs
    iteration_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    @property
    def is_successful(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.final_evaluation.is_satisfactory
    
    def to_summary(self) -> Dict[str, Any]:
        """Generate summary for logging."""
        return {
            "participant_id": self.participant_id,
            "target": self.target_condition,
            "prediction": self.final_prediction.binary_classification.value,
            "probability": self.final_prediction.probability_score,
            "confidence": self.final_prediction.confidence_level.value,
            "satisfactory": self.final_evaluation.is_satisfactory,
            "iterations": self.total_iterations,
            "tokens_used": self.total_tokens_used
        }
