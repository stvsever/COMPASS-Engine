"""
COMPASS Prediction Result Models

Defines the structure for prediction outputs and critic evaluations.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
from datetime import datetime

from .prediction_task import PredictionTaskSpec, PredictionMode

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


class ClassificationPrediction(BaseModel):
    """Node-level classification output."""

    predicted_label: str
    probabilities: Dict[str, float] = Field(default_factory=dict)

    @validator("predicted_label")
    def validate_predicted_label(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("predicted_label must be non-empty")
        return text

    @validator("probabilities")
    def validate_probabilities(cls, value: Dict[str, float]) -> Dict[str, float]:
        probs = {}
        for key, raw in (value or {}).items():
            score = float(raw)
            if score < 0.0 or score > 1.0:
                raise ValueError(f"probability for '{key}' must be in [0,1], got {score}")
            probs[str(key)] = score
        if probs:
            total = sum(probs.values())
            if abs(total - 1.0) > 0.05:
                raise ValueError(f"classification probabilities should sum ~1.0, got {total:.4f}")
        return probs


class RegressionPrediction(BaseModel):
    """Node-level regression output."""

    values: Dict[str, float] = Field(default_factory=dict)

    @validator("values")
    def validate_values(cls, value: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, raw in (value or {}).items():
            out[str(key)] = float(raw)
        if not out:
            raise ValueError("regression values cannot be empty")
        return out


class NodePrediction(BaseModel):
    """Recursive prediction output for one task node."""

    node_id: str
    path: str = "root"
    mode: PredictionMode
    classification: Optional[ClassificationPrediction] = None
    regression: Optional[RegressionPrediction] = None
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_score: float = Field(0.5, ge=0.0, le=1.0)
    key_findings: List[KeyFinding] = Field(default_factory=list)
    reasoning_chain: List[str] = Field(default_factory=list)
    supporting_evidence_for: List[str] = Field(default_factory=list)
    supporting_evidence_against: List[str] = Field(default_factory=list)
    uncertainty_factors: List[str] = Field(default_factory=list)
    children: List["NodePrediction"] = Field(default_factory=list)

    @root_validator(skip_on_failure=True)
    def validate_output_shape(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        mode = values.get("mode")
        cls_pred = values.get("classification")
        reg_pred = values.get("regression")
        if mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION):
            if cls_pred is None:
                raise ValueError(f"classification output required for mode={mode}")
            if reg_pred is not None:
                raise ValueError(f"regression output must be omitted for mode={mode}")
        else:
            if reg_pred is None:
                raise ValueError(f"regression output required for mode={mode}")
            if cls_pred is not None:
                raise ValueError(f"classification output must be omitted for mode={mode}")
        return values

    def walk(self) -> List["NodePrediction"]:
        rows = [self]
        for child in self.children:
            rows.extend(child.walk())
        return rows


NodePrediction.model_rebuild()


class PredictionResult(BaseModel):
    """
    Final prediction result from the Predictor agent.
    
    Supports hierarchical mixed prediction tasks. Binary fields are retained
    as compatibility aliases.
    """
    prediction_id: str = Field(..., description="Unique prediction identifier")
    participant_id: str
    target_condition: str = Field(..., description="Target phenotype label")
    control_condition: str = Field(..., description="Comparator/control label")
    created_at: datetime = Field(default_factory=datetime.now)

    # Canonical generalized prediction outputs
    prediction_task_spec: Optional[PredictionTaskSpec] = None
    root_prediction: Optional[NodePrediction] = None
    flat_predictions: List[NodePrediction] = Field(default_factory=list)

    # Backward-compatible binary aliases
    binary_classification: Optional[BinaryClassification] = None
    probability_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    
    # Evidence
    key_findings: List[KeyFinding] = Field(default_factory=list)
    reasoning_chain: List[str] = Field(default_factory=list)
    supporting_evidence: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "for_case": [],
            "for_control": [],
            "for_target": [],
            "against_target": [],
        }
    )
    uncertainty_factors: List[str] = Field(default_factory=list)
    
    # Summary
    clinical_summary: str = Field("", description="One paragraph clinical summary")
    
    # Execution context
    domains_processed: List[str] = Field(default_factory=list)
    total_tokens_used: int = 0
    iteration: int = 1
    
    @validator("probability_score")
    def validate_probability(cls, v, values):
        """Ensure compatibility probability aligns with binary alias classification."""
        if v is None:
            return v
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

    @root_validator(skip_on_failure=True)
    def finalize_aliases(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        root_prediction: Optional[NodePrediction] = values.get("root_prediction")
        flat_predictions: List[NodePrediction] = list(values.get("flat_predictions") or [])
        task_spec: Optional[PredictionTaskSpec] = values.get("prediction_task_spec")

        if root_prediction and not flat_predictions:
            values["flat_predictions"] = root_prediction.walk()

        # Auto-derive backward-compatible binary aliases when task root is binary.
        if (
            root_prediction
            and values.get("binary_classification") is None
            and root_prediction.mode == PredictionMode.BINARY_CLASSIFICATION
            and root_prediction.classification is not None
        ):
            predicted_label = str(root_prediction.classification.predicted_label or "").strip()
            target_label = ""
            control_label = ""
            if (
                task_spec is not None
                and task_spec.root.mode == PredictionMode.BINARY_CLASSIFICATION
                and len(task_spec.root.class_labels) == 2
            ):
                target_label = str(task_spec.root.class_labels[0] or "").strip()
                control_label = str(task_spec.root.class_labels[1] or "").strip()

            if control_label and predicted_label == control_label:
                values["binary_classification"] = BinaryClassification.CONTROL
            elif target_label and predicted_label == target_label:
                values["binary_classification"] = BinaryClassification.CASE
            else:
                # Legacy fallback for older payloads that still emit CASE/CONTROL text.
                upper_label = predicted_label.upper()
                if "CONTROL" in upper_label:
                    values["binary_classification"] = BinaryClassification.CONTROL
                else:
                    values["binary_classification"] = BinaryClassification.CASE

            if values.get("probability_score") is None:
                probs = root_prediction.classification.probabilities or {}
                if probs:
                    if target_label and target_label in probs:
                        values["probability_score"] = float(probs[target_label])
                    else:
                        case_candidates = [k for k in probs.keys() if "CASE" in str(k).upper() and "CONTROL" not in str(k).upper()]
                        if case_candidates:
                            values["probability_score"] = float(probs[case_candidates[0]])
                        else:
                            # Fallback to predicted label confidence if explicit target key is absent.
                            values["probability_score"] = float(probs.get(predicted_label, 0.5))
                else:
                    values["probability_score"] = 0.5

            values["confidence_level"] = root_prediction.confidence_level

        # Keep target/control consistent with task root when available.
        if task_spec is not None:
            target_label, control_label = task_spec.legacy_target_control()
            if not str(values.get("target_condition") or "").strip():
                values["target_condition"] = target_label
            if not str(values.get("control_condition") or "").strip():
                values["control_condition"] = control_label
        return values

    @property
    def primary_node(self) -> Optional[NodePrediction]:
        return self.root_prediction or (self.flat_predictions[0] if self.flat_predictions else None)

    @property
    def primary_output_kind(self) -> str:
        node = self.primary_node
        if node is None:
            return "unknown"
        if node.mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION):
            return "classification"
        return "regression"
    
    def to_report_dict(self) -> Dict[str, Any]:
        """Generate dictionary suitable for patient report (generalized + legacy fields)."""
        payload: Dict[str, Any] = {
            "participant_id": self.participant_id,
            "condition": self.target_condition,
            "control_condition": self.control_condition,
            "prediction_task_spec": (
                self.prediction_task_spec.model_dump()
                if self.prediction_task_spec is not None and hasattr(self.prediction_task_spec, "model_dump")
                else (self.prediction_task_spec.dict() if self.prediction_task_spec is not None and hasattr(self.prediction_task_spec, "dict") else None)
            ),
            "primary_output_kind": self.primary_output_kind,
            "confidence": self.confidence_level.value,
            "key_findings": [f.finding for f in self.key_findings[:5]],
            "summary": self.clinical_summary,
            "root_prediction": (
                self.root_prediction.model_dump() if self.root_prediction is not None and hasattr(self.root_prediction, "model_dump")
                else (self.root_prediction.dict() if self.root_prediction is not None and hasattr(self.root_prediction, "dict") else None)
            ),
            "flat_predictions": [
                p.model_dump() if hasattr(p, "model_dump") else p.dict() for p in self.flat_predictions
            ],
        }
        if self.binary_classification is not None and self.probability_score is not None:
            payload["prediction"] = self.binary_classification.value
            payload["probability"] = f"{self.probability_score:.2%}"
        return payload


# ============================================================================
# Critic Evaluation
# ============================================================================

class EvaluationChecklist(BaseModel):
    """Checklist of quality criteria (generalized + binary compatibility)."""

    has_required_outputs: bool = False
    output_schema_valid: bool = False
    classification_probabilities_valid: bool = False
    regression_values_valid: bool = False
    hierarchy_consistent: bool = False
    has_binary_outcome: bool = False
    valid_probability: bool = False
    sufficient_coverage: bool = False
    evidence_based_reasoning: bool = False
    clinically_relevant: bool = False
    logically_coherent: bool = False
    critical_domains_processed: bool = False
    active_checks: List[str] = Field(default_factory=list, description="Checklist keys applicable for this run")

    @root_validator(skip_on_failure=True)
    def sync_alias_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Compatibility mappings for legacy binary checklist fields.
        if values.get("classification_probabilities_valid") and not values.get("valid_probability"):
            values["valid_probability"] = True
        if values.get("has_binary_outcome") and not values.get("has_required_outputs"):
            values["has_required_outputs"] = True
        if values.get("valid_probability") and not values.get("classification_probabilities_valid"):
            values["classification_probabilities_valid"] = True
        active = values.get("active_checks") or []
        valid_keys = set(cls._all_generalized_keys())
        cleaned: List[str] = []
        for key in active:
            text = str(key or "").strip()
            if text and text in valid_keys and text not in cleaned:
                cleaned.append(text)
        values["active_checks"] = cleaned
        return values

    @staticmethod
    def _all_generalized_keys() -> List[str]:
        return [
            "has_required_outputs",
            "output_schema_valid",
            "classification_probabilities_valid",
            "regression_values_valid",
            "hierarchy_consistent",
            "sufficient_coverage",
            "evidence_based_reasoning",
            "clinically_relevant",
            "logically_coherent",
            "critical_domains_processed",
        ]

    @property
    def _required_keys(self) -> List[str]:
        if self.active_checks:
            return list(self.active_checks)
        return self._all_generalized_keys()
    
    @property
    def all_passed(self) -> bool:
        """Check if all criteria passed."""
        return all(bool(getattr(self, key, False)) for key in self._required_keys)
    
    @property
    def pass_count(self) -> int:
        """Count of passed criteria."""
        return sum(1 for key in self._required_keys if bool(getattr(self, key, False)))

    @property
    def total_count(self) -> int:
        return len(self._required_keys)


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

    # Fallback metadata (for UI transparency)
    fallback_used: bool = False
    fallback_reason: str = Field("", description="Reason for fallback (e.g., invalid_json)")
    fallback_recommendation: str = Field("", description="Actionable recommendation when fallback is used")
    
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
        prediction_label = (
            self.final_prediction.binary_classification.value
            if self.final_prediction.binary_classification is not None
            else "NON_BINARY"
        )
        return {
            "participant_id": self.participant_id,
            "target": self.target_condition,
            "prediction": prediction_label,
            "probability": self.final_prediction.probability_score,
            "confidence": self.final_prediction.confidence_level.value,
            "satisfactory": self.final_evaluation.is_satisfactory,
            "iterations": self.total_iterations,
            "tokens_used": self.total_tokens_used
        }
