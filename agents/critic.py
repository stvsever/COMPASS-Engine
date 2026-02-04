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
        non_numerical_data: str = None
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
            non_numerical_data
        )
        
        # Call LLM with auto-repair parsing
        evaluation_data = self._call_llm(user_prompt)
        
        # Convert to CriticEvaluation
        evaluation = self._parse_evaluation(evaluation_data, prediction.prediction_id)

        
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
        non_numerical_data: str = None
    ) -> str:
        """Build user prompt for critic evaluation."""
        
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
        
        prompt_parts.extend([
            f"\n## HIERARCHICAL DEVIATION PROFILE (INPUT DATA)",
            f"Note: This is the mean aggregated hierarchy of the multi-modal data. Use this to verify if cited findings exist.",
            str(hierarchical_deviation)[:4000] + "..." if hierarchical_deviation else "Not provided",
            
            f"\n## NON-NUMERICAL CLINICAL NOTES",
            str(non_numerical_data)[:3000] + "..." if non_numerical_data else "Not provided",
            
            f"\n## TARGET CONDITION",
            prediction.target_condition,
            
            f"\n## CONTROL CONDITION",
            "Evaluate whether the 'target phenotype' is present VS whether this data matches better a profile of 'a brain-implicated pathology, but non-psychiatric'!"
            
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
            "- Hallucination (Citing values/findings not in the input data)",
            "- Misinterpretation (Exaggerating low z-scores, misreading values)",
            "- Attribution Error (Attributing findings to wrong domain)",
            "",
            "### 3. COMPLETENESS (Weight: 20%) [BREADTH]",
            "Did the analysis use all available CRITICAL domains?",
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
            reasoning=evaluation_data.get("reasoning", "")
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


from ..config.settings import get_settings

