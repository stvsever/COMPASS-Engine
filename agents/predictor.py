"""
COMPASS Predictor Agent

Synthesizes all processed information into final binary prediction.
"""

import json
import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_agent import BaseAgent
from ..config.settings import get_settings
from ..utils.toon import json_to_toon
from ..utils.token_packer import truncate_text_by_tokens
from ..data.models.prediction_result import (
    PredictionResult,
    KeyFinding,
    BinaryClassification,
    ConfidenceLevel,
)
from ..utils.json_parser import parse_json_response

logger = logging.getLogger("compass.predictor")


class Predictor(BaseAgent):
    """
    The Predictor synthesizes all processed outputs into a final prediction.
    
    Input:
    - Fused outputs from Executor
    - Hierarchical deviation profile (ALWAYS included)
    - Non-numerical data (ALWAYS included)
    
    Output:
    - Binary classification (CASE/CONTROL)
    - Probability score (0.0-1.0)
    - Key findings and reasoning
    """
    
    AGENT_NAME = "Predictor"
    PROMPT_FILE = "predictor_prompt.txt"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings = get_settings()
        
        # Configure LLM params for BaseAgent._call_llm
        self.LLM_MODEL = self.settings.models.predictor_model
        self.LLM_MAX_TOKENS = self.settings.models.predictor_max_tokens
        self.LLM_TEMPERATURE = self.settings.models.predictor_temperature
    
    def execute(
        self,
        executor_output: Dict[str, Any],
        target_condition: str,
        iteration: int = 1
    ) -> PredictionResult:
        """
        Generate prediction from executor outputs.
        
        Args:
            executor_output: Output from Executor agent
            target_condition: "neuropsychiatric" or "neurologic"
            iteration: Which iteration of orchestration loop
        
        Returns:
            PredictionResult with classification and probability
        """
        self._log_start(f"generating {target_condition} prediction")
        
        participant_id = executor_output.get("participant_id", "unknown")
        
        # Build user prompt with all fused data
        user_prompt = self._build_prompt(executor_output, target_condition)
        
        print(f"[Predictor] Participant: {participant_id}")
        print(f"[Predictor] Target condition: {target_condition}")
        print(f"[Predictor] Domains analyzed: {executor_output.get('domains_processed', [])}")
        
        # Call LLM with auto-repair parsing
        prediction_data = self._call_llm(user_prompt)
        
        # Convert to PredictionResult
        result = self._parse_prediction(
            prediction_data=prediction_data,
            participant_id=participant_id,
            target_condition=target_condition,
            executor_output=executor_output,
            iteration=iteration
        )

        
        self._log_complete(
            f"{result.binary_classification.value} (p={result.probability_score:.3f}, "
            f"confidence={result.confidence_level.value})"
        )
        
        # Print prediction summary
        self._print_prediction_summary(result)
        
        return result
    
    def _build_prompt(
        self,
        executor_output: Dict[str, Any],
        target_condition: str
    ) -> str:
        """Build user prompt with all fused data."""
        predictor_input = executor_output.get("predictor_input", {})

        # Token-aware section truncation (prompt budget heuristic).
        max_in = int(getattr(self.settings.token_budget, "max_agent_input_tokens", 20000) or 20000)
        tool_budget = int(max_in * 0.35)
        dev_budget = int(max_in * 0.30)
        mm_budget = int(max_in * 0.45)
        notes_budget = int(max_in * 0.35)
        
        prompt_parts = [
            "## FUSED ANALYSIS OUTPUTS",
            f"\n### Integrated Narrative",
            predictor_input.get("fused_narrative", "No narrative available"),
            
            f"\n### Domain Summaries"
        ]
        
        for domain, summary in predictor_input.get("domain_summaries", {}).items():
            prompt_parts.append(f"- **{domain}**: {summary}")
        
        prompt_parts.extend([
            f"\n### Key Findings",
            json.dumps(predictor_input.get("key_findings", [])[:10], indent=2),
            
            f"\n### Cross-Modal Patterns",
            json.dumps(predictor_input.get("cross_modal_patterns", []), indent=2),
            
            f"\n### Evidence Summary",
            f"For CONTROL: {predictor_input.get('evidence_summary', {}).get('for_control', [])}",
        ])
        
        # Inject RAW TOOL OUTPUTS if available (User Request for transparency)
        raw_tool_outputs = predictor_input.get("tool_outputs_raw")
        if raw_tool_outputs:
            tool_text = json.dumps(raw_tool_outputs, indent=2, default=str)
            tool_text = truncate_text_by_tokens(tool_text, tool_budget, model_hint="gpt-5")
            prompt_parts.extend([
                f"\n## DETAILED TOOL OUTPUTS (RAW - UNFILTERED)",
                f"Use this raw data to confirm findings. Note: Truncated to fit context if very large.",
                tool_text
            ])
            
        dev_raw = predictor_input.get("hierarchical_deviation_raw", predictor_input.get("hierarchical_deviation_summary", "Not available"))
        dev_text = json_to_toon(dev_raw) if isinstance(dev_raw, (dict, list)) else str(dev_raw)
        dev_text = truncate_text_by_tokens(dev_text, dev_budget, model_hint="gpt-5")

        mm_raw = (
            predictor_input.get("multimodal_unprocessed_raw")
            or predictor_input.get("multimodal_context_boost")
            or predictor_input.get("unprocessed_multimodal_data_raw")
        )
        mm_text = json_to_toon(mm_raw) if isinstance(mm_raw, (dict, list)) else str(mm_raw or "Not available")
        mm_text = truncate_text_by_tokens(mm_text, mm_budget, model_hint="gpt-5")

        notes_raw = predictor_input.get("non_numerical_data_raw", predictor_input.get("non_numerical_summary", "Not available"))
        notes_text = truncate_text_by_tokens(str(notes_raw), notes_budget, model_hint="gpt-5")

        data_overview = executor_output.get("data_overview")
        overview_text = ""
        if data_overview:
            overview_text = truncate_text_by_tokens(
                json.dumps(data_overview, indent=2, default=str),
                int(max_in * 0.15),
                model_hint="gpt-5",
            )

        prompt_parts.extend([
            f"\n## DATA OVERVIEW (COVERAGE / TOKENS)",
            overview_text or "Not available",

            f"\n## HIERARCHICAL DEVIATION PROFILE (RAW DATA)",
            # Try raw key first, fall back to summary
            dev_text,
            
            f"\n## MULTIMODAL DATA (CONTEXT - RAW)",
            mm_text,

            
            f"\n## NON-NUMERICAL DATA (CLINICAL NOTES - CRITICAL)",
            # Try raw key first, fall back to summary
            notes_text,
            
            f"\n## TARGET (neuropsychiatric) CONDITION",
            f"Predict: {target_condition}",
            
            f"\n## CONTROL (non-psychiatric) CONDITION",
            "Evaluate whether the 'target phenotype' is present VS whether this data matches better a profile of 'a brain-implicated pathology, but non-psychiatric'!"
            
            "\n## INSTRUCTIONS",
            "Synthesize valid evidence into a binary prediction (CASE or CONTROL).",
            "CRITICAL RULES:",
            "1. CALIBRATED PROBABILITY: Use the full 0.0-1.0 range. Do NOT force false certainty.",
            "   - Ambiguous/Weak Data -> 0.4-0.6 range.",
            "   - Strong Multi-modal Evidence -> >0.8 or <0.2.",
            "2. NON-EXHAUSTIVE REASONING: If data is missing (e.g., no MRI), admit it. Do NOT hallucinate connections.",
            "3. HYPOTHESIS DRIVEN: If findings are subtle, formulate a hypothesis but keep probability conservative.",
            "",
            "",
            "Provide your binary prediction, probability score, and detailed reasoning.",
            "",
            "SMALL NOTE: these participants are from UK Biobank which has an older biological profiles in general ; be aware of this.",
            "",
            "## OUTPUT FORMAT",
            "Return a JSON object with:",
            "{",
            "  \"binary_classification\": \"CASE (likely has target phenotype)\" | \"CONTROL (Likely brain-related implication, however NOT psychiatric profile)\",",
            "  \"probability_score\": float (0.00-1.00),",
            "  \"confidence_level\": \"HIGH\"|\"MEDIUM\"|\"LOW\",",
            "  \"key_findings\": [",
            "    {",
            "      \"domain\": \"domain_name\",",
            "      \"finding\": \"Description\",",
            "      \"direction\": \"ABNORMAL_HIGH\"|\"ABNORMAL_LOW\"|\"NORMAL\",",
            "      \"z_score\": float,",
            "      \"relevance_to_prediction\": \"Explanation\"",
            "    }",
            "  ],",
            "  \"reasoning_chain\": [\"Step 1\", \"Step 2\"],",
            "  \"clinical_summary\": \"Detailed summary\",",
            "  \"supporting_evidence\": {",
            "    \"for_case\": [\"evidence1\"],",
            "    \"for_control\": [\"evidence2\"]",
            "  },",
            "  \"uncertainty_factors\": [\"factor1\"]",
            "}"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_prediction(
        self,
        prediction_data: Dict[str, Any],
        participant_id: str,
        target_condition: str,
        executor_output: Dict[str, Any],
        iteration: int
    ) -> PredictionResult:
        """Parse LLM response into PredictionResult."""
        
        # Parse classification
        # Support long-form research strings and varied LLM outputs
        clean_str = str(prediction_data.get("binary_classification", "CONTROL")).upper()
        if "NOT PSYCHIATRIC" in clean_str or clean_str.startswith("CONTROL"):
            classification = BinaryClassification.CONTROL
        elif "CASE" in clean_str or "TARGET PHENOTYPE" in clean_str:
            classification = BinaryClassification.CASE
        else:
            # Fallback to loose search
            if "CASE" in clean_str:
                classification = BinaryClassification.CASE
            else:
                classification = BinaryClassification.CONTROL
        
        # Parse probability
        probability = prediction_data.get("probability_score", 0.5)
        if isinstance(probability, str):
            try:
                probability = float(probability)
            except ValueError:
                probability = 0.5
        
        # Ensure probability aligns with classification
        if classification == BinaryClassification.CASE and probability < 0.5:
            probability = 0.5 + (0.5 - probability)  # Flip to match
        elif classification == BinaryClassification.CONTROL and probability >= 0.5:
            probability = 0.5 - (probability - 0.5)  # Flip to match
        
        probability = max(0.0, min(1.0, probability))  # Clamp
        
        # Parse confidence
        confidence_str = prediction_data.get("confidence_level", "MEDIUM")
        try:
            confidence = ConfidenceLevel(confidence_str.upper())
        except ValueError:
            confidence = ConfidenceLevel.MEDIUM
        
        # Parse key findings
        key_findings = []
        for finding_data in prediction_data.get("key_findings", [])[:10]:
            if isinstance(finding_data, dict):
                key_findings.append(KeyFinding(
                    domain=finding_data.get("domain", "UNKNOWN"),
                    finding=finding_data.get("finding", ""),
                    direction=finding_data.get("direction", "NORMAL"),
                    z_score=finding_data.get("z_score"),
                    relevance_to_prediction=finding_data.get("relevance_to_prediction", "")
                ))
        
        # Filter out None values from reasoning_chain
        reasoning_chain = [
            step for step in prediction_data.get("reasoning_chain", [])
            if step is not None and isinstance(step, str)
        ]
        
        return PredictionResult(
            prediction_id=prediction_data.get("prediction_id", str(uuid.uuid4())[:8]),
            participant_id=participant_id,
            target_condition=target_condition,
            created_at=datetime.now(),
            binary_classification=classification,
            probability_score=probability,
            confidence_level=confidence,
            key_findings=key_findings,
            reasoning_chain=reasoning_chain,
            supporting_evidence=prediction_data.get("supporting_evidence", {"for_case": [], "for_control": []}),
            uncertainty_factors=prediction_data.get("uncertainty_factors", []),
            clinical_summary=prediction_data.get("clinical_summary", ""),
            domains_processed=executor_output.get("domains_processed", []),
            total_tokens_used=executor_output.get("total_tokens_used", 0),
            iteration=iteration
        )
    
    def _print_prediction_summary(self, result: PredictionResult):
        """Print formatted prediction summary."""
        print(f"\n{'='*60}")
        print(f"PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"Classification: {result.binary_classification.value}")
        print(f"Probability: {result.probability_score:.1%}")
        print(f"Confidence: {result.confidence_level.value}")
        print(f"\nKey Findings:")
        for i, finding in enumerate(result.key_findings[:5], 1):
            print(f"  {i}. [{finding.domain}] {finding.finding[:60]}...")
        print(f"\nClinical Summary:")
        print(f"  {result.clinical_summary[:200]}...")
        print(f"{'='*60}\n")
