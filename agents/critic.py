"""
COMPASS Critic Agent

Evaluates prediction quality and determines if re-orchestration is needed.
"""

import json
import uuid
import logging
from typing import Dict, Any, Optional
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
        print(f"[Critic] Classification: {prediction.binary_classification.value}")
        print(f"[Critic] Probability: {prediction.probability_score:.3f}")
        
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
        
        # Format prediction summary
        prediction_summary = {
            "classification": prediction.binary_classification.value,
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
            "- concise_summary: string (1-2 sentences explaining WHY this verdict was reached)",
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
        
        return "\n".join(prompt_parts)

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
            return parse_json_response(raw_text)
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
            return _attach_fallback(parse_json_response(repaired_raw), "json_repair")
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
            return _attach_fallback(parse_json_response(compact_raw), "compact_reval")
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
        prediction_summary = {
            "classification": prediction.binary_classification.value,
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
        return "\n".join([
            "You are a strict JSON-only evaluator. Output ONLY valid JSON.",
            "Return a minimal evaluation object with keys:",
            "verdict, confidence_in_verdict, composite_score, concise_summary,",
            "score_breakdown {logic,evidence,completeness,relevance},",
            "checklist {has_binary_outcome,valid_probability,sufficient_coverage,evidence_based_reasoning,clinically_relevant,logically_coherent,critical_domains_processed},",
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
        if prediction.binary_classification.value == "CASE":
            valid_probability = valid_probability and float(prediction.probability_score or 0.0) >= 0.5
        if prediction.binary_classification.value == "CONTROL":
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

        checklist = {
            "has_binary_outcome": has_binary_outcome,
            "valid_probability": valid_probability,
            "sufficient_coverage": sufficient_coverage,
            "evidence_based_reasoning": evidence_based_reasoning,
            "clinically_relevant": clinically_relevant,
            "logically_coherent": logically_coherent,
            "critical_domains_processed": critical_domains_processed,
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
        
        # Parse verdict
        verdict_str = evaluation_data.get("verdict", "UNSATISFACTORY")
        try:
            verdict = Verdict(verdict_str.upper())
        except ValueError:
            verdict = Verdict.UNSATISFACTORY
        
        # Parse checklist
        checklist_data = evaluation_data.get("checklist", {})
        checklist = EvaluationChecklist(
            has_binary_outcome=checklist_data.get("has_binary_outcome", False),
            valid_probability=checklist_data.get("valid_probability", False),
            sufficient_coverage=checklist_data.get("sufficient_coverage", False),
            evidence_based_reasoning=checklist_data.get("evidence_based_reasoning", False),
            clinically_relevant=checklist_data.get("clinically_relevant", False),
            logically_coherent=checklist_data.get("logically_coherent", False),
            critical_domains_processed=checklist_data.get("critical_domains_processed", False)
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

        concise_summary = str(evaluation_data.get("concise_summary", "") or "").strip()
        if not concise_summary:
            verdict_text = verdict.value
            composite = float(evaluation_data.get("composite_score", 0.0) or 0.0)
            reasoning = str(evaluation_data.get("reasoning", "") or "").strip()
            if reasoning:
                concise_summary = reasoning[:220].rstrip()
            else:
                concise_summary = f"{verdict_text}: Composite score {composite:.2f}; no concise summary provided by model."
        
        return CriticEvaluation(
            evaluation_id=evaluation_data.get("evaluation_id", str(uuid.uuid4())[:8]),
            prediction_id=prediction_id,
            created_at=datetime.now(),
            verdict=verdict,
            confidence_in_verdict=evaluation_data.get("confidence_in_verdict", 0.5),
            composite_score=evaluation_data.get("composite_score", 0.0),
            score_breakdown=evaluation_data.get("score_breakdown", {}),
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
        checklist = EvaluationChecklist(
            has_binary_outcome=True,
            valid_probability=True,
            sufficient_coverage=False,
            evidence_based_reasoning=False,
            clinically_relevant=False,
            logically_coherent=False,
            critical_domains_processed=False,
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
        
        print(f"\nChecklist ({evaluation.checklist.pass_count}/7):")
        checklist = evaluation.checklist
        status = lambda x: "✓" if x else "✗"
        print(f"  {status(checklist.has_binary_outcome)} Binary outcome")
        print(f"  {status(checklist.valid_probability)} Valid probability")
        print(f"  {status(checklist.sufficient_coverage)} Sufficient coverage")
        print(f"  {status(checklist.evidence_based_reasoning)} Evidence-based reasoning")
        print(f"  {status(checklist.clinically_relevant)} Clinically relevant")
        print(f"  {status(checklist.logically_coherent)} Logically coherent")
        print(f"  {status(checklist.critical_domains_processed)} Critical domains processed")
        
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
