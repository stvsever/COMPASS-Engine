"""
COMPASS Predictor Agent

Synthesizes all processed information into final binary prediction.
Implements no-loss, chunked two-pass reasoning.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import tiktoken

from .base_agent import BaseAgent
from ..config.settings import get_settings
from ..data.models.prediction_result import (
    BinaryClassification,
    ConfidenceLevel,
    KeyFinding,
    PredictionResult,
)
from ..utils.core.multimodal_coverage import feature_key_set
from ..utils.json_parser import parse_json_response
from ..utils.toon import json_to_toon

logger = logging.getLogger("compass.predictor")


class Predictor(BaseAgent):
    """
    The Predictor synthesizes all processed outputs into a final prediction.

    Input:
    - Fused outputs from Executor
    - Hierarchical deviation profile
    - Non-numerical data
    - Coverage ledger

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

        self.LLM_MODEL = self.settings.models.predictor_model
        self.LLM_MAX_TOKENS = self.settings.models.predictor_max_tokens
        self.LLM_TEMPERATURE = self.settings.models.predictor_temperature

        try:
            self._encoder = tiktoken.encoding_for_model(self.LLM_MODEL or "gpt-5")
        except Exception:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    def execute(
        self,
        executor_output: Dict[str, Any],
        target_condition: str,
        iteration: int = 1,
    ) -> PredictionResult:
        """
        Generate prediction from executor outputs using chunked two-pass synthesis.
        """
        self._log_start(f"generating {target_condition} prediction")

        participant_id = executor_output.get("participant_id", "unknown")
        predictor_input = executor_output.get("predictor_input", {}) or {}
        coverage_ledger = (
            executor_output.get("coverage_ledger")
            or predictor_input.get("coverage_ledger")
            or {}
        )
        chunk_evidence = self._require_chunk_evidence(executor_output)

        print(f"[Predictor] Participant: {participant_id}")
        print(f"[Predictor] Target condition: {target_condition}")
        print(f"[Predictor] Domains analyzed: {executor_output.get('domains_processed', [])}")

        coverage_summary = self._validate_feature_representation(
            coverage_ledger=coverage_ledger,
            predictor_input=predictor_input,
            chunk_evidence=chunk_evidence,
        )
        executor_output["coverage_summary"] = coverage_summary

        final_prompt = self._build_final_synthesis_prompt(
            target_condition=target_condition,
            chunk_evidence=chunk_evidence,
            predictor_input=predictor_input,
            executor_output=executor_output,
            coverage_summary=coverage_summary,
        )
        prediction_data = self._call_predictor_json(
            system_prompt=self.system_prompt,
            user_prompt=final_prompt,
            max_retries=2,
        )

        result = self._parse_prediction(
            prediction_data=prediction_data,
            participant_id=participant_id,
            target_condition=target_condition,
            executor_output=executor_output,
            iteration=iteration,
            coverage_summary=coverage_summary,
        )

        self._log_complete(
            f"{result.binary_classification.value} (p={result.probability_score:.3f}, "
            f"confidence={result.confidence_level.value})"
        )
        self._print_prediction_summary(result)
        return result

    def _call_predictor_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        last_error: Optional[str] = None
        prompt = user_prompt
        for attempt in range(max_retries + 1):
            if attempt > 0:
                prompt = (
                    user_prompt
                    + "\n\nPREVIOUS_ERROR:\n"
                    + str(last_error or "unknown")
                    + "\nReturn only fixed valid JSON."
                )
            try:
                response = self.llm_client.call(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.LLM_MODEL,
                    max_tokens=self.LLM_MAX_TOKENS,
                    temperature=self.LLM_TEMPERATURE,
                    response_format={"type": "json_object"},
                )
                self._record_tokens(response.prompt_tokens, response.completion_tokens)
                return parse_json_response(response.content)
            except Exception as e:
                last_error = str(e)
                if attempt == max_retries:
                    raise
        return {}

    def _build_final_synthesis_prompt(
        self,
        *,
        target_condition: str,
        chunk_evidence: List[Dict[str, Any]],
        predictor_input: Dict[str, Any],
        executor_output: Dict[str, Any],
        coverage_summary: Dict[str, Any],
    ) -> str:
        high_priority_context = self._build_high_priority_context(predictor_input, executor_output)
        chunks_text = json_to_toon(chunk_evidence)
        coverage_text = json_to_toon(coverage_summary)

        return "\n".join(
            [
                "Synthesize final CASE/CONTROL verdict from chunk evidence.",
                f"Target condition: {target_condition}",
                "Respect calibration and avoid false positives.",
                "Default to CONTROL if class text is ambiguous.",
                "You MUST integrate all chunk evidence rows.",
                "",
                "## High-priority context (triad)",
                f"```text\n{high_priority_context}\n```",
                "",
                "## Chunk evidence (pass 1 outputs)",
                f"```text\n{chunks_text}\n```",
                "",
                "## Coverage summary",
                f"```text\n{coverage_text}\n```",
                "",
                "Return JSON with fields:",
                "{",
                '  "prediction_id": "string",',
                '  "binary_classification": "CASE|CONTROL|free-text",',
                '  "probability_score": 0.0,',
                '  "confidence_level": "HIGH|MEDIUM|LOW",',
                '  "key_findings": [',
                "    {",
                '      "domain": "domain_name",',
                '      "finding": "Description",',
                '      "direction": "ABNORMAL_HIGH|ABNORMAL_LOW|NORMAL",',
                '      "z_score": float|null,',
                '      "relevance_to_prediction": "Explanation"',
                "    }",
                "  ],",
                '  "reasoning_chain": ["Step 1", "Step 2"],',
                '  "clinical_summary": "Detailed summary",',
                '  "supporting_evidence": {',
                '    "for_case": ["evidence1"],',
                '    "for_control": ["evidence2"]',
                "  },",
                '  "uncertainty_factors": ["factor1"]',
                "}",
            ]
        )

    def _build_high_priority_context(
        self,
        predictor_input: Dict[str, Any],
        executor_output: Dict[str, Any],
    ) -> str:
        non_num = predictor_input.get("non_numerical_data_raw") or executor_output.get("non_numerical_data") or ""
        dev = predictor_input.get("hierarchical_deviation_raw") or executor_output.get("hierarchical_deviation") or {}
        overview = executor_output.get("data_overview") or {}

        def _limit(text: str, max_tokens: int = 2500) -> str:
            tok = self._encoder.encode(text or "")
            return self._encoder.decode(tok[:max_tokens]) if len(tok) > max_tokens else (text or "")

        step_outputs = executor_output.get("step_outputs", {}) or {}

        def _tool_text(tool_name: str) -> str:
            parts = []
            for sid, out in step_outputs.items():
                if not isinstance(out, dict):
                    continue
                name = out.get("tool_name") or (out.get("_step_meta") or {}).get("tool_name")
                if name != tool_name:
                    continue
                parts.append(f"## Step {sid} ({tool_name})\n{json_to_toon(out)}")
            return "\n\n".join(parts) if parts else "Not provided"

        phenotype_text = _tool_text("PhenotypeRepresentation")
        feature_text = _tool_text("FeatureSynthesizer")

        rows = [
            f"## non_numerical_data_raw\n{_limit(str(non_num))}",
            f"## hierarchical_deviation_raw\n{_limit(json_to_toon(dev))}",
            f"## data_overview\n{_limit(json_to_toon(overview))}",
            f"## phenotype_representation\n{_limit(phenotype_text)}",
            f"## feature_synthesizer\n{_limit(feature_text)}",
        ]
        return "\n\n".join(rows)

    def _validate_feature_representation(
        self,
        *,
        coverage_ledger: Dict[str, Any],
        predictor_input: Dict[str, Any],
        chunk_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        all_keys = set(self._as_str_list(coverage_ledger.get("all_features")))
        processed_keys = set(self._as_str_list(coverage_ledger.get("processed_features")))
        unprocessed_raw_keys = feature_key_set(predictor_input.get("multimodal_unprocessed_raw") or {})
        processed_raw_keys = feature_key_set(
            predictor_input.get("multimodal_processed_raw_low_priority") or {}
        )
        cited_keys: Set[str] = set()
        for row in chunk_evidence:
            cited_keys.update(self._as_str_list(row.get("cited_feature_keys")))

        represented = processed_keys | unprocessed_raw_keys | processed_raw_keys | cited_keys
        missing_after_prediction = sorted(list(all_keys - represented))
        summary = {
            "all_feature_count": len(all_keys),
            "processed_feature_count": len(processed_keys),
            "unprocessed_raw_feature_count": len(unprocessed_raw_keys),
            "processed_raw_low_priority_feature_count": len(processed_raw_keys),
            "chunk_cited_feature_count": len(cited_keys),
            "represented_feature_count": len(represented),
            "missing_feature_count": len(missing_after_prediction),
            "missing_features": missing_after_prediction[:2000],
            "invariant_ok": len(missing_after_prediction) == 0,
        }
        if not summary["invariant_ok"]:
            logger.warning(
                "[Predictor] coverage invariant not satisfied after synthesis: %s missing features",
                summary["missing_feature_count"],
            )
        return summary

    def _require_chunk_evidence(self, executor_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunk_evidence = executor_output.get("chunk_evidence")
        if not chunk_evidence:
            raise ValueError("chunk_evidence missing; Integrator must run chunk extraction before Predictor.")
        return chunk_evidence

    def _parse_prediction(
        self,
        prediction_data: Dict[str, Any],
        participant_id: str,
        target_condition: str,
        executor_output: Dict[str, Any],
        iteration: int,
        coverage_summary: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        raw_probability = self._parse_probability(prediction_data.get("probability_score", 0.49))
        classification, ambiguous_class = self._normalize_classification(
            prediction_data.get("binary_classification"),
            raw_probability,
        )
        probability = self._normalize_probability_for_classification(raw_probability, classification)

        confidence_str = prediction_data.get("confidence_level", "MEDIUM")
        try:
            confidence = ConfidenceLevel(str(confidence_str).upper())
        except ValueError:
            confidence = ConfidenceLevel.MEDIUM

        key_findings = []
        for finding_data in prediction_data.get("key_findings", [])[:12]:
            if not isinstance(finding_data, dict):
                continue
            key_findings.append(
                KeyFinding(
                    domain=str(finding_data.get("domain", "UNKNOWN")),
                    finding=str(finding_data.get("finding", "")),
                    direction=str(finding_data.get("direction", "NORMAL")),
                    z_score=finding_data.get("z_score"),
                    relevance_to_prediction=str(
                        finding_data.get("relevance_to_prediction")
                        or finding_data.get("clinical_significance")
                        or ""
                    ),
                )
            )

        reasoning_chain = [
            step for step in prediction_data.get("reasoning_chain", [])
            if step is not None and isinstance(step, str)
        ]

        uncertainty_factors = self._as_str_list(prediction_data.get("uncertainty_factors"))
        if ambiguous_class:
            uncertainty_factors.append("Ambiguous classification text normalized conservatively to CONTROL.")
        if coverage_summary and not coverage_summary.get("invariant_ok", True):
            uncertainty_factors.append(
                f"Coverage invariant warning: {coverage_summary.get('missing_feature_count', 0)} features not represented in evidence."
            )
        uncertainty_factors = self._dedupe_preserve(uncertainty_factors)

        return PredictionResult(
            prediction_id=str(prediction_data.get("prediction_id", str(uuid.uuid4())[:8])),
            participant_id=participant_id,
            target_condition=target_condition,
            created_at=datetime.now(),
            binary_classification=classification,
            probability_score=probability,
            confidence_level=confidence,
            key_findings=key_findings,
            reasoning_chain=reasoning_chain,
            supporting_evidence=prediction_data.get("supporting_evidence", {"for_case": [], "for_control": []}),
            uncertainty_factors=uncertainty_factors,
            clinical_summary=str(prediction_data.get("clinical_summary", "")),
            domains_processed=executor_output.get("domains_processed", []),
            total_tokens_used=executor_output.get("total_tokens_used", 0),
            iteration=iteration,
        )

    def _normalize_classification(
        self,
        raw_classification: Any,
        probability: float,
    ) -> Tuple[BinaryClassification, bool]:
        text = str(raw_classification or "").upper()
        has_case = "CASE" in text or "TARGET PHENOTYPE" in text or "LIKELY HAS TARGET" in text
        has_control = "CONTROL" in text or "NOT PSYCHIATRIC" in text or "NON-PSYCHIATRIC" in text

        if has_case and not has_control:
            return BinaryClassification.CASE, False
        if has_control and not has_case:
            return BinaryClassification.CONTROL, False

        # Ambiguous/missing class text -> conservative default.
        if has_case and has_control:
            return (BinaryClassification.CASE if probability >= 0.5 else BinaryClassification.CONTROL), True
        return BinaryClassification.CONTROL, True

    def _normalize_probability_for_classification(
        self,
        probability: float,
        classification: BinaryClassification,
    ) -> float:
        p = max(0.0, min(1.0, float(probability)))
        # Threshold clamping (no mirror transforms).
        if classification == BinaryClassification.CASE and p < 0.5:
            return 0.5
        if classification == BinaryClassification.CONTROL and p >= 0.5:
            return 0.499
        return p

    def _parse_probability(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            raw = value.strip()
            is_pct = raw.endswith("%")
            if is_pct:
                raw = raw[:-1].strip()
            try:
                out = float(raw)
                if is_pct:
                    out = out / 100.0
                return out
            except ValueError:
                return 0.49
        return 0.49

    def _as_str_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        out = []
        for item in value:
            if item is None:
                continue
            out.append(str(item))
        return out

    def _dedupe_preserve(self, values: List[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for v in values:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    def _print_prediction_summary(self, result: PredictionResult):
        """Print formatted prediction summary."""
        print(f"\n{'='*60}")
        print("PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"Classification: {result.binary_classification.value}")
        print(f"Probability: {result.probability_score:.1%}")
        print(f"Confidence: {result.confidence_level.value}")
        print("\nKey Findings:")
        for i, finding in enumerate(result.key_findings[:5], 1):
            print(f"  {i}. [{finding.domain}] {finding.finding[:60]}...")
        print("\nClinical Summary:")
        print(f"  {result.clinical_summary[:200]}...")
        print(f"{'='*60}\n")
