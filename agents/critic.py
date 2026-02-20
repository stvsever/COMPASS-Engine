"""
COMPASS Critic Agent

Evaluates prediction quality and determines if re-orchestration is needed.
"""

import json
import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_agent import BaseAgent
from ..config.settings import get_settings
from ..data.models.prediction_result import (
    PredictionResult,
    CriticEvaluation,
    EvaluationChecklist,
    ImprovementSuggestion,
    Verdict,
    ImprovementPriority,
)
from ..data.models.execution_plan import PlanExecutionResult
from ..data.models.prediction_task import PredictionTaskSpec, PredictionMode
from ..utils.json_parser import parse_json_response
from ..utils.token_packer import truncate_text_by_tokens
from ..utils.toon import json_to_toon

logger = logging.getLogger("compass.critic")


class Critic(BaseAgent):
    """
    The Critic evaluates prediction quality and determines if it meets standards.
    
    Input:
    - Prediction result from Predictor
    - Execution summary from Executor
    - Original data overview
    
    Output:
    - SATISFACTORY: Prediction passes to output
    - UNSATISFACTORY: Triggers re-orchestration with feedback
    """
    
    AGENT_NAME = "Critic"
    PROMPT_FILE = "critic_prompt.txt"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings = get_settings()
        
        # Configure LLM params for BaseAgent._call_llm
        self.LLM_MODEL = self.settings.models.critic_model
        self.LLM_MAX_TOKENS = self.settings.models.critic_max_tokens
        self.LLM_TEMPERATURE = self.settings.models.critic_temperature
    
    def execute(
        self,
        prediction: PredictionResult,
        executor_output: Dict[str, Any],
        data_overview: Dict[str, Any],
        hierarchical_deviation: Dict[str, Any] = None,
        non_numerical_data: str = None,
        control_condition: Optional[str] = None,
        prediction_task_spec: Optional[PredictionTaskSpec] = None,
    ) -> CriticEvaluation:
        """
        Evaluate a prediction result.
        
        Args:
            prediction: PredictionResult from Predictor
            executor_output: Full output from Executor
            data_overview: Original data overview
            hierarchical_deviation: Full hierarchical deviation map (Input Data)
            non_numerical_data: Clean text of non-numerical notes
        
        Returns:
            CriticEvaluation with verdict and feedback
        """
        self._log_start(f"evaluating prediction {prediction.prediction_id}")

        print(f"[Critic] Evaluating prediction: {prediction.prediction_id}")
        active_task_spec = prediction_task_spec or prediction.prediction_task_spec
        root_mode = (
            active_task_spec.root.mode.value
            if active_task_spec is not None
            else (
                prediction.prediction_task_spec.root.mode.value
                if prediction.prediction_task_spec is not None
                else "unknown"
            )
        )
        print(f"[Critic] Task mode: {root_mode}")
        root = prediction.root_prediction
        if root is not None and root.classification is not None:
            label = str(root.classification.predicted_label or "").strip() or "UNKNOWN"
            print(f"[Critic] Predicted label: {label}")
        if prediction.probability_score is not None:
            print(f"[Critic] Target probability: {prediction.probability_score:.3f}")

        if active_task_spec is not None:
            evaluation = self._evaluate_generalized_prediction(
                prediction=prediction,
                executor_output=executor_output,
                data_overview=data_overview,
                prediction_task_spec=active_task_spec,
            )
            self._log_complete(f"{evaluation.verdict.value} (confidence: {evaluation.confidence_in_verdict:.2f})")
            self._print_evaluation_summary(evaluation)
            return evaluation
        
        # Build user prompt
        user_prompt = self._build_prompt(
            prediction, 
            executor_output, 
            data_overview,
            hierarchical_deviation,
            non_numerical_data,
            control_condition=control_condition,
        )
        
        try:
            # Call LLM and parse with repair fallback (fast-model safe)
            raw = self._call_llm_raw(
                user_prompt,
                max_tokens=self._critic_max_output_tokens(),
                temperature=self.LLM_TEMPERATURE,
            )
            evaluation_data = self._parse_json_with_repair(
                raw,
                prediction=prediction,
                executor_output=executor_output,
                data_overview=data_overview,
                hierarchical_deviation=hierarchical_deviation,
                non_numerical_data=non_numerical_data,
                control_condition=control_condition,
            )
            evaluation = self._parse_evaluation(evaluation_data, prediction.prediction_id)
            if active_task_spec is not None:
                spec_nodes = active_task_spec.node_index()
                required_node_ids = [
                    nid for nid, node in spec_nodes.items() if bool(getattr(node, "required", True))
                ]
                flat_nodes = prediction.flat_predictions or (
                    prediction.root_prediction.walk() if prediction.root_prediction is not None else []
                )
                pred_node_ids = {str(node.node_id) for node in flat_nodes}
                missing_required_nodes = [nid for nid in required_node_ids if nid not in pred_node_ids]
                concise_summary = self._build_generalized_concise_summary(
                    prediction_task_spec=active_task_spec,
                    verdict=evaluation.verdict,
                    composite_score=float(evaluation.composite_score),
                    checklist=evaluation.checklist,
                    primary_output_summary=self._summarize_primary_output(prediction),
                    required_node_count=len(required_node_ids),
                    missing_required_nodes=missing_required_nodes,
                    fallback_used=bool(getattr(evaluation, "fallback_used", False)),
                )
                if not active_task_spec.is_pure_binary_root():
                    concise_summary = self._polish_generalized_concise_summary(
                        base_summary=concise_summary,
                        prediction_task_spec=active_task_spec,
                        verdict=evaluation.verdict,
                        composite_score=float(evaluation.composite_score),
                        checklist=evaluation.checklist,
                        primary_output_summary=self._summarize_primary_output(prediction),
                        missing_required_nodes=missing_required_nodes,
                        fallback_used=bool(getattr(evaluation, "fallback_used", False)),
                    )
                evaluation.concise_summary = concise_summary
        except Exception as e:
            logger.exception("Critic evaluation failed; returning deterministic UNSAT fallback.")
            self._log_error(f"LLM/JSON failure, using fallback evaluation: {e}")
            evaluation = self._build_fallback_evaluation(
                prediction_id=prediction.prediction_id,
                error=str(e),
            )

        
        self._log_complete(f"{evaluation.verdict.value} (confidence: {evaluation.confidence_in_verdict:.2f})")
        
        # Print evaluation summary
        self._print_evaluation_summary(evaluation)
        
        return evaluation

    def _evaluate_generalized_prediction(
        self,
        *,
        prediction: PredictionResult,
        executor_output: Dict[str, Any],
        data_overview: Dict[str, Any],
        prediction_task_spec: PredictionTaskSpec,
    ) -> CriticEvaluation:
        """Deterministic generalized evaluator for non-binary task modes."""
        flat_nodes = prediction.flat_predictions or (
            prediction.root_prediction.walk() if prediction.root_prediction is not None else []
        )
        pred_nodes = {str(n.node_id): n for n in flat_nodes}
        spec_nodes = prediction_task_spec.node_index()

        has_classification_nodes = any(
            node.mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION)
            for node in spec_nodes.values()
        )
        has_regression_nodes = any(
            node.mode in (PredictionMode.UNIVARIATE_REGRESSION, PredictionMode.MULTIVARIATE_REGRESSION)
            for node in spec_nodes.values()
        )
        has_hierarchy = len(spec_nodes) > 1 or any(bool(node.children) for node in spec_nodes.values())

        required_node_ids = [nid for nid, node in spec_nodes.items() if bool(getattr(node, "required", True))]
        missing_required_nodes = [nid for nid in required_node_ids if nid not in pred_nodes]
        has_required_outputs = len(missing_required_nodes) == 0
        unexpected_nodes = [nid for nid in pred_nodes.keys() if nid not in spec_nodes]
        hierarchy_relevant = has_hierarchy or bool(unexpected_nodes)

        classification_probabilities_valid = True
        regression_values_valid = True
        regression_default_suspect_nodes: List[str] = []
        hierarchy_consistent = True

        for node_id, node_spec in spec_nodes.items():
            pred_node = pred_nodes.get(node_id)
            if pred_node is None:
                if bool(getattr(node_spec, "required", True)):
                    hierarchy_consistent = False
                continue

            if node_spec.mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION):
                cls_pred = pred_node.classification
                if cls_pred is None:
                    classification_probabilities_valid = False
                    continue
                probs = cls_pred.probabilities or {}
                if not probs:
                    classification_probabilities_valid = False
                    continue
                total = sum(float(v) for v in probs.values())
                if abs(total - 1.0) > 0.05:
                    classification_probabilities_valid = False
                if cls_pred.predicted_label not in list(node_spec.class_labels):
                    classification_probabilities_valid = False
            else:
                reg_pred = pred_node.regression
                if reg_pred is None:
                    regression_values_valid = False
                    continue
                values = reg_pred.values or {}
                node_values: List[float] = []
                for output_name in node_spec.regression_outputs:
                    if output_name not in values:
                        regression_values_valid = False
                        continue
                    try:
                        numeric_value = float(values[output_name])
                        node_values.append(numeric_value)
                        if not (numeric_value == numeric_value and abs(numeric_value) != float("inf")):
                            regression_values_valid = False
                    except Exception:
                        regression_values_valid = False
                if node_values and all(abs(v) <= 1e-12 for v in node_values):
                    regression_default_suspect_nodes.append(node_id)

        zero_default_unjustified = False
        if regression_default_suspect_nodes:
            narrative_blob = " ".join(
                [str(prediction.clinical_summary or "")]
                + [str(step or "") for step in list(prediction.reasoning_chain or [])]
            ).lower()
            zero_markers = (
                "near zero",
                "approximately zero",
                "around zero",
                "close to zero",
                "no deviation",
                "at mean",
                "neutral baseline",
                "minimal effect",
            )
            zero_default_unjustified = not any(marker in narrative_blob for marker in zero_markers)
            if zero_default_unjustified:
                regression_values_valid = False

        if unexpected_nodes:
            hierarchy_consistent = False

        mode_checks_valid = True
        if has_classification_nodes:
            mode_checks_valid = mode_checks_valid and classification_probabilities_valid
        if has_regression_nodes:
            mode_checks_valid = mode_checks_valid and regression_values_valid
        if hierarchy_relevant:
            mode_checks_valid = mode_checks_valid and hierarchy_consistent
        output_schema_valid = has_required_outputs and mode_checks_valid

        domains_processed = list(executor_output.get("domains_processed") or [])
        domain_coverage = data_overview.get("domain_coverage") if isinstance(data_overview, dict) else {}
        available_domains = []
        if isinstance(domain_coverage, dict):
            for domain, cov in domain_coverage.items():
                if isinstance(cov, dict) and int(cov.get("present_leaves", 0) or 0) > 0:
                    available_domains.append(str(domain))
        available_count = len(available_domains)
        if available_count > 0:
            processed_count = len(set(domains_processed) & set(available_domains))
            coverage_ratio = processed_count / float(available_count)
        else:
            coverage_ratio = 1.0
        sufficient_coverage = coverage_ratio >= 0.6

        uncertainty_flags = [str(x or "") for x in list(getattr(prediction, "uncertainty_factors", []) or [])]
        fallback_used = (
            str(getattr(prediction, "prediction_id", "") or "").startswith("fallback_")
            or any("fallback" in flag.lower() for flag in uncertainty_flags)
            or "deterministic fallback" in str(prediction.clinical_summary or "").lower()
        )
        evidence_based_reasoning = bool(prediction.key_findings) and bool(prediction.reasoning_chain)
        clinically_relevant = bool(str(prediction.clinical_summary or "").strip())
        logically_coherent = bool(prediction.reasoning_chain or prediction.clinical_summary)
        critical_domains_processed = len(domains_processed) > 0
        if fallback_used:
            # Fallback outputs are schema-preserving safety outputs, not final trustworthy inferences.
            evidence_based_reasoning = False
            clinically_relevant = False
            logically_coherent = False
        has_binary_outcome = prediction.binary_classification is not None
        valid_probability = (
            prediction.probability_score is not None
            and 0.0 <= float(prediction.probability_score) <= 1.0
        )

        active_checks: List[str] = [
            "has_required_outputs",
            "output_schema_valid",
        ]
        if has_classification_nodes:
            active_checks.append("classification_probabilities_valid")
        if has_regression_nodes:
            active_checks.append("regression_values_valid")
        if hierarchy_relevant:
            active_checks.append("hierarchy_consistent")
        active_checks.extend(
            [
                "sufficient_coverage",
                "evidence_based_reasoning",
                "clinically_relevant",
                "logically_coherent",
                "critical_domains_processed",
            ]
        )

        checklist = EvaluationChecklist(
            has_required_outputs=has_required_outputs,
            output_schema_valid=output_schema_valid,
            classification_probabilities_valid=classification_probabilities_valid,
            regression_values_valid=regression_values_valid,
            hierarchy_consistent=hierarchy_consistent,
            has_binary_outcome=has_binary_outcome,
            valid_probability=valid_probability,
            sufficient_coverage=sufficient_coverage,
            evidence_based_reasoning=evidence_based_reasoning,
            clinically_relevant=clinically_relevant,
            logically_coherent=logically_coherent,
            critical_domains_processed=critical_domains_processed,
            active_checks=active_checks,
        )

        hierarchy_term = float(hierarchy_consistent) if hierarchy_relevant else 1.0
        logic_score = (float(logically_coherent) + float(output_schema_valid) + hierarchy_term) / 3.0
        evidence_score = (
            float(evidence_based_reasoning)
            + float(mode_checks_valid)
        ) / 2.0
        completeness_score = (
            float(has_required_outputs)
            + float(sufficient_coverage)
            + float(critical_domains_processed)
        ) / 3.0
        relevance_score = float(clinically_relevant)

        score_breakdown = {
            "logic": round(logic_score, 4),
            "evidence": round(evidence_score, 4),
            "completeness": round(completeness_score, 4),
            "relevance": round(relevance_score, 4),
        }
        composite_score = (
            score_breakdown["logic"] * 0.40
            + score_breakdown["evidence"] * 0.30
            + score_breakdown["completeness"] * 0.20
            + score_breakdown["relevance"] * 0.10
        )
        verdict = Verdict.SATISFACTORY if (
            composite_score >= 0.70
            and output_schema_valid
            and has_required_outputs
            and not fallback_used
        ) else Verdict.UNSATISFACTORY
        confidence_in_verdict = min(0.95, max(0.4, checklist.pass_count / float(max(1, checklist.total_count))))

        strengths: List[str] = []
        weaknesses: List[str] = []
        domains_missed: List[str] = []
        improvements: List[ImprovementSuggestion] = []

        if output_schema_valid:
            strengths.append("Prediction output structure matches requested task modes.")
        else:
            weaknesses.append("Prediction output schema is incomplete or inconsistent with task specification.")
            improvements.append(
                ImprovementSuggestion(
                    issue="Output schema mismatch",
                    suggestion="Ensure each required task node includes mode-consistent prediction fields.",
                    priority=ImprovementPriority.HIGH,
                )
            )

        if sufficient_coverage:
            strengths.append("Available data domains were sufficiently processed.")
        else:
            weaknesses.append("Coverage of available data domains is below threshold.")
            improvements.append(
                ImprovementSuggestion(
                    issue="Insufficient domain coverage",
                    suggestion="Process additional high-value domains before final synthesis.",
                    priority=ImprovementPriority.HIGH,
                )
            )
            domains_missed = [d for d in available_domains if d not in set(domains_processed)]

        if missing_required_nodes:
            weaknesses.append(f"Missing required prediction nodes: {', '.join(missing_required_nodes)}")
            improvements.append(
                ImprovementSuggestion(
                    issue="Missing required task nodes",
                    suggestion="Generate outputs for every required node in the hierarchy.",
                    priority=ImprovementPriority.HIGH,
                )
            )

        if unexpected_nodes:
            weaknesses.append(f"Prediction included undefined task nodes: {', '.join(unexpected_nodes)}")
            improvements.append(
                ImprovementSuggestion(
                    issue="Unexpected task nodes",
                    suggestion="Emit predictions only for node_ids defined in the task specification.",
                    priority=ImprovementPriority.MEDIUM,
                )
            )

        if evidence_based_reasoning:
            strengths.append("Reasoning includes evidence-backed findings.")
        else:
            weaknesses.append("Reasoning lacks sufficient evidence grounding.")
            improvements.append(
                ImprovementSuggestion(
                    issue="Weak evidence grounding",
                    suggestion="Reference explicit findings for each major conclusion.",
                    priority=ImprovementPriority.MEDIUM,
                )
            )
        if zero_default_unjustified:
            weaknesses.append(
                "Regression outputs appear to be template defaults (all zeros) without explicit justification."
            )
            improvements.append(
                ImprovementSuggestion(
                    issue="Template-like regression defaults",
                    suggestion=(
                        "Provide concrete non-default numeric estimates or explicitly justify near-zero outputs in the reasoning."
                    ),
                    priority=ImprovementPriority.HIGH,
                )
            )
        if fallback_used:
            weaknesses.append("Predictor fallback response detected; final outputs are not reliable for sign-off.")
            improvements.append(
                ImprovementSuggestion(
                    issue="Predictor fallback used",
                    suggestion="Rerun with stricter prompt compliance or a higher-capability predictor model.",
                    priority=ImprovementPriority.HIGH,
                )
            )

        primary_output_summary = self._summarize_primary_output(prediction)
        concise_summary = self._build_generalized_concise_summary(
            prediction_task_spec=prediction_task_spec,
            verdict=verdict,
            composite_score=composite_score,
            checklist=checklist,
            primary_output_summary=primary_output_summary,
            required_node_count=len(required_node_ids),
            missing_required_nodes=missing_required_nodes,
            fallback_used=fallback_used,
        )
        concise_summary = self._polish_generalized_concise_summary(
            base_summary=concise_summary,
            prediction_task_spec=prediction_task_spec,
            verdict=verdict,
            composite_score=composite_score,
            checklist=checklist,
            primary_output_summary=primary_output_summary,
            missing_required_nodes=missing_required_nodes,
            fallback_used=fallback_used,
        )
        reasoning = (
            f"Required nodes present={has_required_outputs}; output schema valid={output_schema_valid}; "
            f"classification probabilities valid={classification_probabilities_valid if has_classification_nodes else 'n/a'}; "
            f"regression values valid={regression_values_valid if has_regression_nodes else 'n/a'}; "
            f"hierarchy consistent={hierarchy_consistent if hierarchy_relevant else 'n/a'}; "
            f"coverage ratio={coverage_ratio:.2f}; fallback_used={fallback_used}."
        )

        return CriticEvaluation(
            evaluation_id=str(uuid.uuid4())[:8],
            prediction_id=prediction.prediction_id,
            created_at=datetime.now(),
            verdict=verdict,
            confidence_in_verdict=float(confidence_in_verdict),
            composite_score=float(round(composite_score, 4)),
            score_breakdown=score_breakdown,
            checklist=checklist,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=improvements,
            domains_missed=domains_missed,
            reasoning=reasoning,
            concise_summary=concise_summary,
        )

    def _summarize_primary_output(self, prediction: PredictionResult) -> str:
        root = prediction.root_prediction
        if root is None:
            return "not available"
        if root.mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION):
            cls = root.classification
            if cls is None:
                return "classification unavailable"
            label = str(cls.predicted_label or "UNKNOWN")
            score = None
            if isinstance(cls.probabilities, dict):
                score = cls.probabilities.get(label)
                if score is None and cls.probabilities:
                    try:
                        score = max(float(v) for v in cls.probabilities.values())
                    except Exception:
                        score = None
            if isinstance(score, (int, float)):
                return f"{label} ({float(score):.2f})"
            return label
        reg = root.regression
        if reg is None or not isinstance(reg.values, dict) or not reg.values:
            return "regression unavailable"
        entries: List[str] = []
        for key, value in list(reg.values.items())[:4]:
            try:
                entries.append(f"{key}={float(value):.3f}")
            except Exception:
                entries.append(f"{key}={value}")
        if len(reg.values) > 4:
            entries.append(f"+{len(reg.values) - 4} more")
        return "; ".join(entries)

    def _build_generalized_concise_summary(
        self,
        *,
        prediction_task_spec: PredictionTaskSpec,
        verdict: Verdict,
        composite_score: float,
        checklist: EvaluationChecklist,
        primary_output_summary: str,
        required_node_count: int,
        missing_required_nodes: List[str],
        fallback_used: bool,
    ) -> str:
        mode = prediction_task_spec.root.mode
        is_hierarchical = len(prediction_task_spec.node_index()) > 1
        mode_name = {
            PredictionMode.BINARY_CLASSIFICATION: "Binary classification",
            PredictionMode.MULTICLASS_CLASSIFICATION: "Multi-class classification",
            PredictionMode.UNIVARIATE_REGRESSION: "Univariate regression",
            PredictionMode.MULTIVARIATE_REGRESSION: "Multivariate regression",
        }.get(mode, str(mode.value).replace("_", " ").title())
        passed = verdict == Verdict.SATISFACTORY
        checklist_text = f"{checklist.pass_count}/{checklist.total_count}"
        state_text = "passed quality checks" if passed else "needs revision"
        rationale_text = (
            "Required output, schema, evidence, and coverage checks were satisfied."
            if passed
            else "One or more required schema, evidence, or coverage checks failed."
        )
        fallback_text = " Predictor fallback was detected; rerun is recommended." if fallback_used else ""

        if is_hierarchical:
            mode_title = f"Hierarchical task ({mode_name} root)"
            if missing_required_nodes:
                node_text = (
                    f"Required node coverage incomplete ({required_node_count - len(missing_required_nodes)}/"
                    f"{required_node_count}); missing: {', '.join(missing_required_nodes[:3])}."
                )
            else:
                node_text = f"Required node coverage complete ({required_node_count}/{required_node_count})."
            return (
                f"{mode_title} {state_text} ({checklist_text} checks, score {composite_score:.2f}). "
                f"{node_text} Root output: {primary_output_summary}. {rationale_text}{fallback_text}"
            )

        if mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION):
            output_label = "Predicted label"
        elif mode == PredictionMode.UNIVARIATE_REGRESSION:
            output_label = "Estimated value"
        else:
            output_label = "Estimated profile"
        return (
            f"{mode_name} {state_text} ({checklist_text} checks, score {composite_score:.2f}). "
            f"{output_label}: {primary_output_summary}. {rationale_text}{fallback_text}"
        )

    def _polish_generalized_concise_summary(
        self,
        *,
        base_summary: str,
        prediction_task_spec: PredictionTaskSpec,
        verdict: Verdict,
        composite_score: float,
        checklist: EvaluationChecklist,
        primary_output_summary: str,
        missing_required_nodes: List[str],
        fallback_used: bool,
    ) -> str:
        """Use Critic LLM style generation for clearer run-summary text, with safe deterministic fallback."""
        if not hasattr(self.llm_client, "call"):
            return base_summary
        try:
            payload = {
                "mode": prediction_task_spec.root.mode.value,
                "hierarchical": len(prediction_task_spec.node_index()) > 1,
                "verdict": verdict.value,
                "composite_score": round(float(composite_score), 3),
                "checks_passed": int(checklist.pass_count),
                "checks_total": int(checklist.total_count),
                "primary_output": primary_output_summary,
                "missing_required_nodes": list(missing_required_nodes),
                "fallback_used": bool(fallback_used),
            }
            prompt = "\n".join(
                [
                    "Rewrite this evaluator snapshot into a clear user-facing run summary.",
                    "Requirements:",
                    "- English only.",
                    "- 1 to 2 sentences.",
                    "- Mention task mode, primary output, and why verdict was reached.",
                    "- If fallback_used=true, explicitly recommend rerun.",
                    "- Avoid jargon and avoid placeholder wording.",
                    "",
                    "Snapshot JSON:",
                    json.dumps(payload, ensure_ascii=False),
                    "",
                    'Return strict JSON: {"concise_summary":"..."}',
                ]
            )
            raw = self._call_llm_raw(
                prompt,
                model=self.LLM_MODEL or self.settings.models.critic_model,
                max_tokens=220,
                temperature=0.1,
                expect_json=True,
                system_prompt="You are a precise evaluator summary writer. Return only JSON.",
            )
            parsed = parse_json_response(raw, expected_keys=["concise_summary"])
            text = str(parsed.get("concise_summary") or "").strip()
            if not text:
                return base_summary
            text = " ".join(text.split())
            if len(text) > 320:
                text = text[:317].rstrip() + "..."
            return text
        except Exception:
            return base_summary
    
    def _build_prompt(
        self,
        prediction: PredictionResult,
        executor_output: Dict[str, Any],
        data_overview: Dict[str, Any],
        hierarchical_deviation: Dict[str, Any] = None,
        non_numerical_data: str = None,
        control_condition: Optional[str] = None,
    ) -> str:
        """Build user prompt for critic evaluation."""

        max_in = int(getattr(self.settings.token_budget, "max_agent_input_tokens", 30000) or 30000)
        pred_input_budget = int(max_in * 0.38)
        dev_budget = int(max_in * 0.25)
        notes_budget = int(max_in * 0.22)
        dataflow_budget = int(max_in * 0.15)
        
        root = prediction.root_prediction
        root_label = ""
        if root is not None and root.classification is not None:
            root_label = str(root.classification.predicted_label or "").strip()
        classification_text = (
            prediction.binary_classification.value
            if prediction.binary_classification is not None
            else (root_label or "NON_BINARY")
        )

        # Format prediction summary
        prediction_summary = {
            "classification": classification_text,
            "probability": prediction.probability_score,
            "confidence": prediction.confidence_level.value,
            "key_findings": [
                {"domain": f.domain, "finding": f.finding}
                for f in prediction.key_findings[:5]
            ],
            "reasoning_chain": prediction.reasoning_chain[:5],
            "clinical_summary": prediction.clinical_summary
        }
        
        # Format execution summary
        exec_result = executor_output.get("execution_result")
        if isinstance(exec_result, PlanExecutionResult):
            execution_summary = {
                "steps_completed": exec_result.steps_completed,
                "steps_failed": exec_result.steps_failed,
                "tokens_used": exec_result.total_tokens_used,
                "errors": exec_result.errors
            }
        else:
            execution_summary = {"status": "unknown"}
        
        # Format domain coverage
        coverage_summary = {}
        if "domain_coverage" in data_overview:
            for domain, cov in data_overview["domain_coverage"].items():
                coverage_summary[domain] = {
                    "coverage": cov.get("coverage_percentage", 0),
                    "present": cov.get("present_leaves", 0)
                }
        
        prompt_parts = [
            "## PREDICTION RESULT TO EVALUATE",
            f"```json\n{json.dumps(prediction_summary, indent=2)}\n```",
            
            f"\n## EXECUTION SUMMARY",
            f"```json\n{json.dumps(execution_summary, indent=2)}\n```",
            
            f"\n## ORIGINAL DATA OVERVIEW",
            f"Domains available: {executor_output.get('domains_processed', [])}",
            f"Coverage by domain:",
        ]
        
        for domain, cov in coverage_summary.items():
            prompt_parts.append(f"  - {domain}: {cov['coverage']:.1f}%")

        # Provide the Critic with the actual fused input used by the Predictor (evidence traceability).
        predictor_input = executor_output.get("predictor_input", {}) or {}
        predictor_input_text = truncate_text_by_tokens(
            json_to_toon(predictor_input),
            pred_input_budget,
            model_hint="gpt-5",
        )
        dataflow_summary = executor_output.get("dataflow_summary") or {}
        dataflow_text = truncate_text_by_tokens(
            json_to_toon(dataflow_summary),
            dataflow_budget,
            model_hint="gpt-5",
        )
        prompt_parts.extend([
            f"\n## PREDICTOR INPUT (EVIDENCE SNAPSHOT)",
            f"Use this to verify whether cited findings are present in the provided context.",
            f"```text\n{predictor_input_text}\n```",
        ])
        if dataflow_summary:
            prompt_parts.extend([
                f"\n## DATAFLOW SUMMARY & ASSERTIONS",
                f"Objective coverage/chunking/context-fill status for this iteration.",
                f"```text\n{dataflow_text}\n```",
            ])
        
        ctrl = (
            getattr(prediction, "control_condition", None)
            or control_condition
            or executor_output.get("control_condition")
            or "Control condition not provided"
        )

        prompt_parts.extend([
            f"\n## HIERARCHICAL DEVIATION PROFILE (INPUT DATA)",
            f"Note: This is the mean aggregated hierarchy of the multi-modal data (so there is no direction; only means 'abnormal' without necesarilly implying pathology ; Use this to verify if cited findings exist. The actual multi-modal data is NOT always given to you; just a compressed summary.",
            truncate_text_by_tokens(
                json_to_toon(hierarchical_deviation) if hierarchical_deviation else "Not provided",
                dev_budget,
                model_hint="gpt-5",
            ),
            
            f"\n## NON-NUMERICAL CLINICAL NOTES",
            truncate_text_by_tokens(str(non_numerical_data) if non_numerical_data else "Not provided", notes_budget, model_hint="gpt-5"),
            
            f"\n## TARGET CONDITION",
            prediction.target_condition,
            
            f"\n## CONTROL CONDITION",
            f"Evaluate whether the 'target phenotype' is present (case) VS whether this data matches better a profile of: {ctrl}",
            
            "\n## EVALUATION TASK",
            "Evaluate this prediction using the following Hierarchical Multi-composite Satisfaction Scoring Matrix.",
            "You must calculate a 'score' (0.00-1.00) for each component and a final weighted 'composite_score'.",
            "",
            "### 1. LOGICAL COHERENCE (Weight: 40%) [CRITICAL]",
            "Does the reasoning follow a sound logical progression?",
            "CHECK FOR THESE ERRORS:",
            "- Circular Reasoning (e.g., 'It is X because it is X')",
            "- Non-Sequitur (Conclusion does not follow from premises)",
            "- Contradiction (Conflicting statements in reasoning)",
            "- Ignored Counter-Evidence (Ignoring 'Normal' findings that rule out the condition)",
            "- Hasty Generalization (Predicting CASE based on weak evidence)",
            "",
            "### 2. EVIDENCE VERIFICATION (Weight: 30%) [FOUNDATION]",
            "Do the cited findings actually exist in the provided Input Data?",
            "CHECK FOR THESE ERRORS:",
            "- Hallucination (Citing values that are not plausible given the data_overview; you will not be given ALL raw leaf-level input data)",
            "- Misinterpretation (Exaggerating z-scores, small effects, irrelevant findings as highly predictive, misreading values)",
            "IMPORTANT: you are NOT passed all the raw mulit-modal data so you can not fully verify each leaf-node level finding. ..."
            "",
            "### 3. COMPLETENESS (Weight: 20%) [BREADTH]",
            "Did the analysis use all available CRITICAL (i.e., truly high useful) domains?",
            "- Penalty if available MRI/Genomic data was ignored.",
            "- Penalty for failing to report 'Normal' findings (crucial for differential diagnosis).",
            "",
            "### 4. CLINICAL RELEVANCE (Weight: 10%) [UTILITY]",
            "Are findings specific to the target condition?",
            "- Penalty for generic statements applicable to any patient.",
            "",
            "## OUTPUT FORMAT",
            "Return a JSON object with:",
            "- verdict: 'SATISFACTORY' or 'UNSATISFACTORY'",
            "- confidence_in_verdict: float (0-1)",
            "- composite_score: float (0.00-1.00)",
            "- concise_summary: string (1-2 sentences in professional English explaining WHY this verdict was reached)",
            "- score_breakdown: { 'logic': float, 'evidence': float, 'completeness': float, 'relevance': float }",
            "- checklist: {",
            "    'has_binary_outcome': bool,",
            "    'valid_probability': bool,",
            "    'sufficient_coverage': bool,",
            "    'evidence_based_reasoning': bool,",
            "    'clinically_relevant': bool,",
            "    'logically_coherent': bool,",
            "    'critical_domains_processed': bool",
            "  }",
            "- improvement_suggestions: List of specific fixes if score < 1.0",
            "- reasoning: Detailed explanation of the scoring deductions",
            "",
            "Ensure feedback is comprehensive and actionable."
        ])
        
        return self._append_runtime_instruction(
            "\n".join(prompt_parts),
            label="Critic Runtime Instruction",
        )

    def _critic_max_output_tokens(self) -> int:
        max_agent_out = int(getattr(self.settings.token_budget, "max_agent_output_tokens", 16000) or 16000)
        if self.LLM_MAX_TOKENS:
            return min(int(self.LLM_MAX_TOKENS), max_agent_out)
        return max_agent_out

    def _call_llm_raw(
        self,
        user_prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        expect_json: bool = True,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call LLM and return raw text (allows custom JSON repair)."""
        model = model or self.LLM_MODEL or self.settings.models.tool_model
        max_tokens = max_tokens or self._critic_max_output_tokens()
        temperature = temperature if temperature is not None else (self.LLM_TEMPERATURE or 0.0)

        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = {"messages": messages, "model": model, "max_tokens": max_tokens, "temperature": temperature}
        if expect_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.llm_client.call(**kwargs)
        self._record_tokens(response.prompt_tokens, response.completion_tokens)
        return response.content

    def _parse_json_with_repair(
        self,
        raw_text: str,
        *,
        prediction: PredictionResult,
        executor_output: Dict[str, Any],
        data_overview: Dict[str, Any],
        hierarchical_deviation: Dict[str, Any],
        non_numerical_data: str,
        control_condition: Optional[str],
    ) -> Dict[str, Any]:
        """Parse JSON with LLM repair + compact fallback to avoid UNSAT on fast models."""
        expected_keys = [
            "verdict",
            "confidence_in_verdict",
            "composite_score",
            "concise_summary",
            "score_breakdown",
            "checklist",
            "improvement_suggestions",
            "reasoning",
        ]

        def _attach_fallback(data: Dict[str, Any], reason: str) -> Dict[str, Any]:
            data = dict(data or {})
            data["fallback_used"] = True
            data["fallback_reason"] = reason
            data["fallback_recommendation"] = (
                "Critic used fallback parsing due to invalid JSON output. "
                "Strongly recommend a higher-quality critic model for reliable evaluations."
            )
            return data

        try:
            parsed = parse_json_response(raw_text, expected_keys=expected_keys)
            return self._normalize_evaluation_payload(
                parsed,
                prediction=prediction,
                executor_output=executor_output,
                data_overview=data_overview,
                hierarchical_deviation=hierarchical_deviation,
                non_numerical_data=non_numerical_data,
                control_condition=control_condition,
            )
        except Exception as first_err:
            logger.warning("Critic JSON parse failed; attempting repair: %s", first_err)

        # 1) LLM-based JSON repair using tool model (small prompt, strict JSON)
        try:
            truncated = truncate_text_by_tokens(raw_text, 4000, model_hint="gpt-5")
            repair_prompt = (
                "You are a JSON repair utility. Convert the INPUT into valid JSON.\n"
                "Rules:\n"
                "- Return ONLY valid JSON (no markdown, no commentary).\n"
                "- Preserve keys/values when possible; add missing keys with reasonable defaults.\n"
                "- Ensure strings are properly escaped.\n\n"
                "INPUT:\n"
                f"{truncated}\n"
            )
            repaired_raw = self._call_llm_raw(
                repair_prompt,
                model=self.settings.models.tool_model,
                max_tokens=1200,
                temperature=0.0,
                expect_json=True,
                system_prompt="You are a strict JSON repair utility. Return ONLY valid JSON.",
            )
            repaired = parse_json_response(repaired_raw, expected_keys=expected_keys)
            normalized = self._normalize_evaluation_payload(
                repaired,
                prediction=prediction,
                executor_output=executor_output,
                data_overview=data_overview,
                hierarchical_deviation=hierarchical_deviation,
                non_numerical_data=non_numerical_data,
                control_condition=control_condition,
            )
            return _attach_fallback(normalized, "json_repair")
        except Exception as repair_err:
            logger.warning("Critic JSON repair failed; attempting compact re-eval: %s", repair_err)

        # 2) Compact re-evaluation prompt (lower output complexity)
        compact_prompt = self._build_compact_prompt(
            prediction,
            executor_output,
            data_overview,
            hierarchical_deviation,
            non_numerical_data,
            control_condition=control_condition,
        )
        compact_raw = self._call_llm_raw(
            compact_prompt,
            model=self.settings.models.tool_model,
            max_tokens=1200,
            temperature=0.0,
            expect_json=True,
        )
        try:
            compact = parse_json_response(compact_raw, expected_keys=expected_keys)
            normalized = self._normalize_evaluation_payload(
                compact,
                prediction=prediction,
                executor_output=executor_output,
                data_overview=data_overview,
                hierarchical_deviation=hierarchical_deviation,
                non_numerical_data=non_numerical_data,
                control_condition=control_condition,
            )
            return _attach_fallback(normalized, "compact_reval")
        except Exception as compact_err:
            logger.warning("Critic compact JSON parse failed; falling back to heuristic evaluation: %s", compact_err)
            return _attach_fallback(self._heuristic_evaluation_data(
                prediction=prediction,
                executor_output=executor_output,
                data_overview=data_overview,
                hierarchical_deviation=hierarchical_deviation,
                non_numerical_data=non_numerical_data,
                control_condition=control_condition,
            ), "heuristic")

    def _normalize_evaluation_payload(
        self,
        parsed: Dict[str, Any],
        *,
        prediction: PredictionResult,
        executor_output: Dict[str, Any],
        data_overview: Dict[str, Any],
        hierarchical_deviation: Dict[str, Any],
        non_numerical_data: str,
        control_condition: Optional[str],
    ) -> Dict[str, Any]:
        """
        Make critic payload robust when models return partial JSON.
        Uses heuristic evaluation as baseline and overlays model-provided fields.
        """
        baseline = self._heuristic_evaluation_data(
            prediction=prediction,
            executor_output=executor_output,
            data_overview=data_overview,
            hierarchical_deviation=hierarchical_deviation,
            non_numerical_data=non_numerical_data,
            control_condition=control_condition,
        )

        if not isinstance(parsed, dict):
            return baseline

        def _as_bool(value: Any, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                low = value.strip().lower()
                if low in {"true", "yes", "y", "1", "pass", "passed"}:
                    return True
                if low in {"false", "no", "n", "0", "fail", "failed"}:
                    return False
            return default

        def _as_float01(value: Any, default: float) -> float:
            try:
                v = float(value)
            except Exception:
                return float(default)
            return max(0.0, min(1.0, v))

        out = dict(baseline)

        verdict_raw = str(parsed.get("verdict") or "").strip().upper()
        if verdict_raw in {"SATISFACTORY", "UNSATISFACTORY"}:
            out["verdict"] = verdict_raw

        checklist = dict(out.get("checklist") or {})
        parsed_checklist = parsed.get("checklist")
        if isinstance(parsed_checklist, dict):
            for key in (
                "has_required_outputs",
                "output_schema_valid",
                "classification_probabilities_valid",
                "regression_values_valid",
                "hierarchy_consistent",
                "has_binary_outcome",
                "valid_probability",
                "sufficient_coverage",
                "evidence_based_reasoning",
                "clinically_relevant",
                "logically_coherent",
                "critical_domains_processed",
            ):
                if key in parsed_checklist:
                    checklist[key] = _as_bool(parsed_checklist.get(key), bool(checklist.get(key, False)))
            parsed_active_checks = parsed_checklist.get("active_checks")
            if isinstance(parsed_active_checks, list):
                cleaned_active = []
                for key in parsed_active_checks:
                    text = str(key or "").strip()
                    if text and text not in cleaned_active:
                        cleaned_active.append(text)
                checklist["active_checks"] = cleaned_active
        out["checklist"] = checklist

        score_breakdown = dict(out.get("score_breakdown") or {})
        parsed_breakdown = parsed.get("score_breakdown")
        if isinstance(parsed_breakdown, dict):
            for key in ("logic", "evidence", "completeness", "relevance"):
                if key in parsed_breakdown:
                    score_breakdown[key] = _as_float01(parsed_breakdown.get(key), float(score_breakdown.get(key, 0.0)))
        out["score_breakdown"] = score_breakdown

        if "composite_score" in parsed:
            out["composite_score"] = _as_float01(parsed.get("composite_score"), float(out.get("composite_score", 0.0)))
        else:
            # Derive composite from score breakdown if model omitted it.
            logic = _as_float01(score_breakdown.get("logic"), 0.0)
            evidence = _as_float01(score_breakdown.get("evidence"), 0.0)
            completeness = _as_float01(score_breakdown.get("completeness"), 0.0)
            relevance = _as_float01(score_breakdown.get("relevance"), 0.0)
            out["composite_score"] = round(
                0.4 * logic + 0.3 * evidence + 0.2 * completeness + 0.1 * relevance,
                2,
            )

        out["confidence_in_verdict"] = _as_float01(
            parsed.get("confidence_in_verdict"),
            float(out.get("confidence_in_verdict", 0.5)),
        )

        if isinstance(parsed.get("strengths"), list):
            out["strengths"] = [str(x).strip() for x in parsed.get("strengths", []) if str(x).strip()][:8]
        if isinstance(parsed.get("weaknesses"), list):
            out["weaknesses"] = [str(x).strip() for x in parsed.get("weaknesses", []) if str(x).strip()][:10]

        if isinstance(parsed.get("domains_missed"), list):
            out["domains_missed"] = [str(x).strip() for x in parsed.get("domains_missed", []) if str(x).strip()][:12]

        if isinstance(parsed.get("improvement_suggestions"), list):
            cleaned = []
            for row in parsed.get("improvement_suggestions", []):
                if not isinstance(row, dict):
                    continue
                issue = str(row.get("issue") or "").strip()
                suggestion = str(row.get("suggestion") or "").strip()
                priority = str(row.get("priority") or "MEDIUM").upper().strip()
                if priority not in {"HIGH", "MEDIUM", "LOW"}:
                    priority = "MEDIUM"
                if issue or suggestion:
                    cleaned.append({"issue": issue, "suggestion": suggestion, "priority": priority})
            if cleaned:
                out["improvement_suggestions"] = cleaned

        reasoning = str(parsed.get("reasoning") or "").strip()
        if reasoning:
            out["reasoning"] = reasoning

        concise = str(parsed.get("concise_summary") or parsed.get("summary") or "").strip()
        if concise:
            out["concise_summary"] = concise
        else:
            failed = [k for k, v in checklist.items() if isinstance(v, bool) and not bool(v)]
            fail_txt = ", ".join(failed[:3]) if failed else "none"
            verdict_txt = str(out.get("verdict") or "UNSATISFACTORY")
            comp = float(out.get("composite_score") or 0.0)
            out["concise_summary"] = f"{verdict_txt}: Composite score {comp:.2f}; failed criteria: {fail_txt}."

        if str(out.get("verdict") or "").upper() not in {"SATISFACTORY", "UNSATISFACTORY"}:
            out["verdict"] = (
                "SATISFACTORY"
                if (
                    float(out.get("composite_score", 0.0)) >= 0.7
                    and bool(checklist.get("evidence_based_reasoning"))
                    and bool(checklist.get("logically_coherent"))
                )
                else "UNSATISFACTORY"
            )

        out["evaluation_id"] = str(parsed.get("evaluation_id") or out.get("evaluation_id") or str(uuid.uuid4())[:8])
        return out

    def _build_compact_prompt(
        self,
        prediction: PredictionResult,
        executor_output: Dict[str, Any],
        data_overview: Dict[str, Any],
        hierarchical_deviation: Dict[str, Any],
        non_numerical_data: str,
        control_condition: Optional[str] = None,
    ) -> str:
        """Smaller critic prompt for fast/unstable JSON models."""
        root = prediction.root_prediction
        root_label = ""
        if root is not None and root.classification is not None:
            root_label = str(root.classification.predicted_label or "").strip()
        classification_text = (
            prediction.binary_classification.value
            if prediction.binary_classification is not None
            else (root_label or "NON_BINARY")
        )

        prediction_summary = {
            "classification": classification_text,
            "probability": prediction.probability_score,
            "confidence": prediction.confidence_level.value,
            "key_findings": [
                {"domain": f.domain, "finding": f.finding}
                for f in prediction.key_findings[:5]
            ],
            "clinical_summary": prediction.clinical_summary[:800],
        }
        coverage_summary = {}
        if "domain_coverage" in data_overview:
            for domain, cov in data_overview["domain_coverage"].items():
                coverage_summary[domain] = {
                    "coverage": cov.get("coverage_percentage", 0),
                    "present": cov.get("present_leaves", 0),
                }
        ctrl = (
            getattr(prediction, "control_condition", None)
            or control_condition
            or executor_output.get("control_condition")
            or "Control condition not provided"
        )
        notes_snippet = truncate_text_by_tokens(
            str(non_numerical_data or "Not provided"), 800, model_hint="gpt-5"
        )
        deviation_snippet = truncate_text_by_tokens(
            json_to_toon(hierarchical_deviation) if hierarchical_deviation else "Not provided",
            800,
            model_hint="gpt-5",
        )
        prompt = "\n".join([
            "You are a strict JSON-only evaluator. Output ONLY valid JSON.",
            "Return a minimal evaluation object with keys:",
            "verdict, confidence_in_verdict, composite_score, concise_summary,",
            "score_breakdown {logic,evidence,completeness,relevance},",
            "checklist {has_binary_outcome,valid_probability,sufficient_coverage,evidence_based_reasoning,clinically_relevant,logically_coherent,critical_domains_processed},",
            "Use professional English for concise_summary and reasoning.",
            "improvement_suggestions (optional list), reasoning (short).",
            "",
            "## PREDICTION",
            json.dumps(prediction_summary, indent=2),
            "",
            "## DOMAINS",
            f"processed: {executor_output.get('domains_processed', [])}",
            f"coverage: {coverage_summary}",
            "",
            "## NOTES (SNIPPET)",
            notes_snippet,
            "",
            "## HIERARCHICAL DEVIATION (SNIPPET)",
            deviation_snippet,
            "",
            "## TARGET",
            prediction.target_condition,
            "",
            "## CONTROL",
            ctrl,
        ])
        return self._append_runtime_instruction(prompt, label="Critic Runtime Instruction")

    def _heuristic_evaluation_data(
        self,
        *,
        prediction: PredictionResult,
        executor_output: Dict[str, Any],
        data_overview: Dict[str, Any],
        hierarchical_deviation: Dict[str, Any],
        non_numerical_data: str,
        control_condition: Optional[str],
    ) -> Dict[str, Any]:
        """Deterministic evaluation fallback using available structured data."""
        available_domains = []
        for domain, cov in (data_overview.get("domain_coverage", {}) or {}).items():
            if cov.get("present_leaves", 0) > 0 or cov.get("coverage_percentage", 0) > 0:
                available_domains.append(domain)

        processed = executor_output.get("domains_processed") or prediction.domains_processed or []
        processed_set = set(str(d) for d in processed)
        available_set = set(str(d) for d in available_domains)

        coverage_ratio = 1.0
        if available_set:
            coverage_ratio = len(available_set & processed_set) / max(1, len(available_set))

        sufficient_coverage = (coverage_ratio >= 0.7) if available_set else True

        critical = ["BRAIN_MRI", "GENOMICS", "BIOLOGICAL_ASSAY", "COGNITION"]
        present_critical = [d for d in critical if d in available_set]
        critical_domains_processed = all(d in processed_set for d in present_critical) if present_critical else True

        has_binary_outcome = bool(getattr(prediction, "binary_classification", None))
        valid_probability = 0.0 <= float(prediction.probability_score or 0.0) <= 1.0
        cls_value = str(getattr(getattr(prediction, "binary_classification", None), "value", "") or "").upper()
        if cls_value == "CASE":
            valid_probability = valid_probability and float(prediction.probability_score or 0.0) >= 0.5
        if cls_value == "CONTROL":
            valid_probability = valid_probability and float(prediction.probability_score or 0.0) < 0.5

        key_domains = [str(f.domain) for f in (prediction.key_findings or []) if getattr(f, "domain", None)]
        evidence_based_reasoning = bool(key_domains) and all(d in available_set for d in key_domains)
        logically_coherent = bool(prediction.reasoning_chain) or bool(prediction.clinical_summary)
        clinically_relevant = bool(prediction.key_findings) and bool(prediction.clinical_summary)

        logic_score = 1.0 if logically_coherent else 0.0
        evidence_score = 1.0 if evidence_based_reasoning else 0.0
        completeness_score = min(1.0, max(0.0, coverage_ratio))
        relevance_score = 1.0 if clinically_relevant else 0.0

        composite_score = (
            0.4 * logic_score
            + 0.3 * evidence_score
            + 0.2 * completeness_score
            + 0.1 * relevance_score
        )

        has_required_outputs = has_binary_outcome
        classification_probabilities_valid = valid_probability
        output_schema_valid = has_required_outputs and classification_probabilities_valid
        active_checks = [
            "has_required_outputs",
            "output_schema_valid",
            "classification_probabilities_valid",
            "sufficient_coverage",
            "evidence_based_reasoning",
            "clinically_relevant",
            "logically_coherent",
            "critical_domains_processed",
        ]

        checklist = {
            "has_required_outputs": has_required_outputs,
            "output_schema_valid": output_schema_valid,
            "classification_probabilities_valid": classification_probabilities_valid,
            "has_binary_outcome": has_binary_outcome,
            "valid_probability": valid_probability,
            "sufficient_coverage": sufficient_coverage,
            "evidence_based_reasoning": evidence_based_reasoning,
            "clinically_relevant": clinically_relevant,
            "logically_coherent": logically_coherent,
            "critical_domains_processed": critical_domains_processed,
            "active_checks": active_checks,
        }

        improvement_suggestions = []
        if not sufficient_coverage:
            improvement_suggestions.append({
                "issue": "Insufficient domain coverage processed",
                "suggestion": "Ensure all available domains are passed to the predictor and referenced in reasoning.",
                "priority": "HIGH",
            })
        if not evidence_based_reasoning:
            improvement_suggestions.append({
                "issue": "Key findings not clearly grounded in available data",
                "suggestion": "Cite only findings present in predictor input snapshot and domain coverage.",
                "priority": "HIGH",
            })
        if not clinically_relevant:
            improvement_suggestions.append({
                "issue": "Clinical relevance unclear",
                "suggestion": "Link findings explicitly to target phenotype rather than generic statements.",
                "priority": "MEDIUM",
            })
        if not logically_coherent:
            improvement_suggestions.append({
                "issue": "Reasoning chain missing or unclear",
                "suggestion": "Provide a short, logically ordered reasoning chain based on data.",
                "priority": "MEDIUM",
            })
        if not critical_domains_processed and present_critical:
            improvement_suggestions.append({
                "issue": "Critical domains not processed",
                "suggestion": f"Include critical domains: {', '.join(present_critical)}.",
                "priority": "HIGH",
            })

        verdict = "SATISFACTORY" if (composite_score >= 0.7 and evidence_based_reasoning and logically_coherent) else "UNSATISFACTORY"
        concise_summary = (
            f"Heuristic evaluation applied due to JSON failure. Coverage={coverage_ratio:.2f}, "
            f"evidence_based={evidence_based_reasoning}, coherent={logically_coherent}."
        )
        domains_missed = [d for d in available_domains if d not in processed_set]

        return {
            "evaluation_id": str(uuid.uuid4())[:8],
            "verdict": verdict,
            "confidence_in_verdict": 0.4 if verdict == "UNSATISFACTORY" else 0.6,
            "composite_score": round(composite_score, 2),
            "concise_summary": concise_summary,
            "score_breakdown": {
                "logic": round(logic_score, 2),
                "evidence": round(evidence_score, 2),
                "completeness": round(completeness_score, 2),
                "relevance": round(relevance_score, 2),
            },
            "checklist": checklist,
            "improvement_suggestions": improvement_suggestions,
            "domains_missed": domains_missed,
            "reasoning": concise_summary,
        }

    
    def _parse_evaluation(
        self,
        evaluation_data: Dict[str, Any],
        prediction_id: str
    ) -> CriticEvaluation:
        """Parse LLM response into CriticEvaluation."""
        def _as_bool(value: Any, default: bool = False) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                low = value.strip().lower()
                if low in {"true", "yes", "y", "1", "pass", "passed"}:
                    return True
                if low in {"false", "no", "n", "0", "fail", "failed"}:
                    return False
            return default

        def _as_float01(value: Any, default: float) -> float:
            try:
                v = float(value)
            except Exception:
                return float(default)
            return max(0.0, min(1.0, v))
        
        # Parse verdict
        verdict_str = evaluation_data.get("verdict", "UNSATISFACTORY")
        try:
            verdict = Verdict(verdict_str.upper())
        except ValueError:
            verdict = Verdict.UNSATISFACTORY
        
        # Parse checklist
        checklist_data = evaluation_data.get("checklist", {})
        if not isinstance(checklist_data, dict):
            checklist_data = {}
        active_checks_raw = checklist_data.get("active_checks")
        if not isinstance(active_checks_raw, list):
            active_checks_raw = evaluation_data.get("active_checks")
        active_checks = [str(k).strip() for k in list(active_checks_raw or []) if str(k).strip()]

        has_binary_fields = ("has_binary_outcome" in checklist_data) or ("valid_probability" in checklist_data)
        has_generalized_fields = any(
            key in checklist_data
            for key in (
                "has_required_outputs",
                "output_schema_valid",
                "classification_probabilities_valid",
                "regression_values_valid",
                "hierarchy_consistent",
            )
        )
        inferred_binary_mode = has_binary_fields and not has_generalized_fields

        has_required_outputs = _as_bool(
            checklist_data.get("has_required_outputs"),
            _as_bool(checklist_data.get("has_binary_outcome"), False),
        )
        classification_probabilities_valid = _as_bool(
            checklist_data.get("classification_probabilities_valid"),
            _as_bool(checklist_data.get("valid_probability"), False),
        )
        output_schema_valid = _as_bool(
            checklist_data.get("output_schema_valid"),
            has_required_outputs and classification_probabilities_valid,
        )
        regression_values_valid = _as_bool(checklist_data.get("regression_values_valid"), False)
        hierarchy_consistent = _as_bool(checklist_data.get("hierarchy_consistent"), False)

        if not active_checks and inferred_binary_mode:
            active_checks = [
                "has_required_outputs",
                "output_schema_valid",
                "classification_probabilities_valid",
                "sufficient_coverage",
                "evidence_based_reasoning",
                "clinically_relevant",
                "logically_coherent",
                "critical_domains_processed",
            ]
        elif not active_checks and has_generalized_fields:
            active_checks = [
                "has_required_outputs",
                "output_schema_valid",
            ]
            if "classification_probabilities_valid" in checklist_data:
                active_checks.append("classification_probabilities_valid")
            if "regression_values_valid" in checklist_data:
                active_checks.append("regression_values_valid")
            if "hierarchy_consistent" in checklist_data:
                active_checks.append("hierarchy_consistent")
            active_checks.extend(
                [
                    "sufficient_coverage",
                    "evidence_based_reasoning",
                    "clinically_relevant",
                    "logically_coherent",
                    "critical_domains_processed",
                ]
            )
        checklist = EvaluationChecklist(
            has_required_outputs=has_required_outputs,
            output_schema_valid=output_schema_valid,
            classification_probabilities_valid=classification_probabilities_valid,
            regression_values_valid=regression_values_valid,
            hierarchy_consistent=hierarchy_consistent,
            has_binary_outcome=_as_bool(checklist_data.get("has_binary_outcome", False), False),
            valid_probability=_as_bool(checklist_data.get("valid_probability", False), False),
            sufficient_coverage=_as_bool(checklist_data.get("sufficient_coverage", False), False),
            evidence_based_reasoning=_as_bool(checklist_data.get("evidence_based_reasoning", False), False),
            clinically_relevant=_as_bool(checklist_data.get("clinically_relevant", False), False),
            logically_coherent=_as_bool(checklist_data.get("logically_coherent", False), False),
            critical_domains_processed=_as_bool(checklist_data.get("critical_domains_processed", False), False),
            active_checks=active_checks,
        )
        
        # Parse improvement suggestions
        suggestions = []
        for sugg_data in evaluation_data.get("improvement_suggestions", []):
            if isinstance(sugg_data, dict):
                priority_str = sugg_data.get("priority", "MEDIUM")
                try:
                    priority = ImprovementPriority(priority_str.upper())
                except ValueError:
                    priority = ImprovementPriority.MEDIUM
                
                suggestions.append(ImprovementSuggestion(
                    issue=sugg_data.get("issue", ""),
                    suggestion=sugg_data.get("suggestion", ""),
                    priority=priority
                ))

        concise_summary = str(evaluation_data.get("concise_summary", "") or evaluation_data.get("summary", "") or "").strip()
        if not concise_summary:
            verdict_text = verdict.value
            composite = _as_float01(evaluation_data.get("composite_score", 0.0), 0.0)
            reasoning = str(evaluation_data.get("reasoning", "") or "").strip()
            if reasoning:
                concise_summary = reasoning[:220].rstrip()
            else:
                concise_summary = f"{verdict_text}: Composite score {composite:.2f}; no concise summary provided by model."

        score_breakdown = evaluation_data.get("score_breakdown", {})
        if isinstance(score_breakdown, dict):
            score_breakdown = {
                k: _as_float01(v, 0.0)
                for k, v in score_breakdown.items()
                if isinstance(k, str)
            }
        else:
            score_breakdown = {}
        
        return CriticEvaluation(
            evaluation_id=evaluation_data.get("evaluation_id", str(uuid.uuid4())[:8]),
            prediction_id=prediction_id,
            created_at=datetime.now(),
            verdict=verdict,
            confidence_in_verdict=_as_float01(evaluation_data.get("confidence_in_verdict", 0.5), 0.5),
            composite_score=_as_float01(evaluation_data.get("composite_score", 0.0), 0.0),
            score_breakdown=score_breakdown,
            checklist=checklist,
            strengths=evaluation_data.get("strengths", []),
            weaknesses=evaluation_data.get("weaknesses", []),
            improvement_suggestions=suggestions,
            domains_missed=evaluation_data.get("domains_missed", []),
            reasoning=evaluation_data.get("reasoning", ""),
            concise_summary=concise_summary,
            fallback_used=bool(evaluation_data.get("fallback_used", False) or evaluation_data.get("_fallback_used", False)),
            fallback_reason=str(evaluation_data.get("fallback_reason", "") or ""),
            fallback_recommendation=str(evaluation_data.get("fallback_recommendation", "") or ""),
        )

    def _build_fallback_evaluation(self, prediction_id: str, error: str) -> CriticEvaluation:
        """Create deterministic fail-safe evaluation when critic LLM output is invalid."""
        active_checks = [
            "has_required_outputs",
            "output_schema_valid",
            "classification_probabilities_valid",
            "sufficient_coverage",
            "evidence_based_reasoning",
            "clinically_relevant",
            "logically_coherent",
            "critical_domains_processed",
        ]
        checklist = EvaluationChecklist(
            has_required_outputs=True,
            output_schema_valid=True,
            classification_probabilities_valid=True,
            has_binary_outcome=True,
            valid_probability=True,
            sufficient_coverage=False,
            evidence_based_reasoning=False,
            clinically_relevant=False,
            logically_coherent=False,
            critical_domains_processed=False,
            active_checks=active_checks,
        )
        suggestion = ImprovementSuggestion(
            issue="Critic output was not machine-parseable JSON",
            suggestion=(
                "Retry with a stricter JSON-capable model/provider or reduce critic prompt complexity. "
                "Prediction is preserved, but this attempt is marked UNSATISFACTORY."
            ),
            priority=ImprovementPriority.HIGH,
        )
        reasoning = (
            "Critic LLM response could not be parsed as valid JSON after retries. "
            f"Raw error: {error}"
        )
        return CriticEvaluation(
            evaluation_id=str(uuid.uuid4())[:8],
            prediction_id=prediction_id,
            created_at=datetime.now(),
            verdict=Verdict.UNSATISFACTORY,
            confidence_in_verdict=0.0,
            composite_score=0.0,
            score_breakdown={
                "logic": 0.0,
                "evidence": 0.0,
                "completeness": 0.0,
                "relevance": 0.0,
            },
            checklist=checklist,
            strengths=[],
            weaknesses=[
                "Critic output parsing failed",
                "Evaluation reliability unavailable for this iteration",
            ],
            improvement_suggestions=[suggestion],
            domains_missed=[],
            reasoning=reasoning,
            concise_summary=(
                "Critic response was invalid JSON; applied deterministic UNSAT fallback. "
                "Pipeline continues without crashing."
            ),
            fallback_used=True,
            fallback_reason="deterministic_fallback",
            fallback_recommendation=(
                "Critic used deterministic fallback due to invalid JSON output. "
                "Strongly recommend a higher-quality critic model for reliable evaluations."
            ),
        )

    
    def _print_evaluation_summary(self, evaluation: CriticEvaluation):
        """Print formatted evaluation summary."""
        verdict_symbol = "✓" if evaluation.is_satisfactory else "✗"
        
        print(f"\n{'='*60}")
        print(f"CRITIC EVALUATION")
        print(f"{'='*60}")
        print(f"Verdict: {verdict_symbol} {evaluation.verdict.value}")
        print(f"Confidence: {evaluation.confidence_in_verdict:.1%}")
        print(f"Composite Score: {evaluation.composite_score:.2f} / 1.00")
        
        if evaluation.score_breakdown:
            print(f"Scoring Breakdown:")
            for category, score in evaluation.score_breakdown.items():
                print(f"  - {category.title()}: {score:.2f}")
        
        print(f"\nChecklist ({evaluation.checklist.pass_count}/{evaluation.checklist.total_count}):")
        checklist = evaluation.checklist
        status = lambda x: "✓" if x else "✗"
        ordered_checks = [
            ("has_required_outputs", "Required outputs present"),
            ("output_schema_valid", "Output schema valid"),
            ("classification_probabilities_valid", "Classification probabilities valid"),
            ("regression_values_valid", "Regression values valid"),
            ("hierarchy_consistent", "Hierarchy consistent"),
            ("sufficient_coverage", "Sufficient coverage"),
            ("evidence_based_reasoning", "Evidence-based reasoning"),
            ("clinically_relevant", "Clinically relevant"),
            ("logically_coherent", "Logically coherent"),
            ("critical_domains_processed", "Critical domains processed"),
        ]
        active = set(checklist.active_checks or [key for key, _ in ordered_checks])
        for key, label in ordered_checks:
            if key in active:
                print(f"  {status(bool(getattr(checklist, key, False)))} {label}")
        
        if evaluation.strengths:
            print(f"\nStrengths:")
            for s in evaluation.strengths[:3]:
                print(f"  + {s[:60]}...")
        
        if evaluation.weaknesses:
            print(f"\nWeaknesses:")
            for w in evaluation.weaknesses[:3]:
                print(f"  - {w[:60]}...")
        
        if not evaluation.is_satisfactory and evaluation.improvement_suggestions:
            print(f"\nImprovement Suggestions:")
            for sugg in evaluation.high_priority_issues[:3]:
                print(f"  [{sugg.priority.value}] {sugg.issue}: {sugg.suggestion[:50]}...")
        
        print(f"{'='*60}\n")
