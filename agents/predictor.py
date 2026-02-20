"""
COMPASS Predictor Agent

Synthesizes all processed information into final phenotype prediction outputs.
Implements no-loss, chunked two-pass reasoning.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import tiktoken

from .base_agent import BaseAgent
from ..config.settings import get_settings, LLMBackend
from ..data.models.prediction_result import (
    BinaryClassification,
    ClassificationPrediction,
    ConfidenceLevel,
    KeyFinding,
    NodePrediction,
    PredictionResult,
    RegressionPrediction,
)
from ..data.models.prediction_task import (
    PredictionMode,
    PredictionTaskNode,
    PredictionTaskSpec,
    build_binary_task_spec,
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
    - Node-wise classification/regression outputs per task spec
    - Optional binary compatibility aliases (when root task is binary)
    - Key findings and reasoning
    """

    AGENT_NAME = "Predictor"
    PROMPT_FILE = "predictor_prompt.txt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings = get_settings()
        self._prompt_compaction_meta: Dict[str, Any] = {}
        self._active_prediction_task_spec: Optional[PredictionTaskSpec] = None

        self.LLM_MODEL = self.settings.models.predictor_model
        self.LLM_MAX_TOKENS = self.settings.models.predictor_max_tokens
        self.LLM_TEMPERATURE = self.settings.models.predictor_temperature

        try:
            self._encoder = tiktoken.encoding_for_model(self.LLM_MODEL or "gpt-5")
        except Exception:
            try:
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoder = None

    def execute(
        self,
        executor_output: Dict[str, Any],
        target_condition: str,
        control_condition: str,
        prediction_task_spec: Optional[PredictionTaskSpec] = None,
        iteration: int = 1,
    ) -> PredictionResult:
        """
        Generate prediction from executor outputs using chunked two-pass synthesis.
        """
        if prediction_task_spec is None:
            prediction_task_spec = build_binary_task_spec(
                target_label=target_condition,
                control_label=control_condition,
            )
        self._active_prediction_task_spec = prediction_task_spec
        self._log_start(f"generating {target_condition} prediction")

        participant_id = executor_output.get("participant_id", "unknown")
        predictor_input = executor_output.get("predictor_input", {}) or {}
        coverage_ledger = (
            executor_output.get("coverage_ledger")
            or predictor_input.get("coverage_ledger")
            or {}
        )
        chunk_evidence = executor_output.get("chunk_evidence") or []
        chunking_skipped = bool(executor_output.get("chunking_skipped"))
        if not chunk_evidence and not chunking_skipped:
            raise ValueError("chunk_evidence missing; Integrator must run chunk extraction before Predictor.")

        print(f"[Predictor] Participant: {participant_id}")
        print(f"[Predictor] Target condition: {target_condition}")
        if prediction_task_spec.root.mode in (
            PredictionMode.BINARY_CLASSIFICATION,
            PredictionMode.MULTICLASS_CLASSIFICATION,
        ) and str(control_condition or "").strip():
            print(f"[Predictor] Control condition: {control_condition}")
        print(f"[Predictor] Domains analyzed: {executor_output.get('domains_processed', [])}")

        coverage_summary = self._validate_feature_representation(
            coverage_ledger=coverage_ledger,
            predictor_input=predictor_input,
            chunk_evidence=chunk_evidence,
        )
        executor_output["coverage_summary"] = coverage_summary
        high_priority_context = self._build_high_priority_context(predictor_input, executor_output)
        self._prompt_compaction_meta = {}

        if chunking_skipped:
            non_core_context = executor_output.get("non_core_context_text") or "Not provided"
            if prediction_task_spec.is_pure_binary_root():
                final_prompt = self._build_direct_synthesis_prompt(
                    target_condition=target_condition,
                    control_condition=control_condition,
                    predictor_input=predictor_input,
                    executor_output=executor_output,
                    coverage_summary=coverage_summary,
                    high_priority_context=high_priority_context,
                    non_core_context=non_core_context,
                )
            else:
                final_prompt = self._build_generalized_synthesis_prompt(
                    prediction_task_spec=prediction_task_spec,
                    target_condition=target_condition,
                    control_condition=control_condition,
                    high_priority_context=high_priority_context,
                    chunk_evidence=[],
                    coverage_summary=coverage_summary,
                    non_core_context=non_core_context,
                    chunking_skipped=True,
                )
            predictor_call_context = {
                "mode": "direct",
                "high_priority_context": high_priority_context,
                "non_core_context": non_core_context,
                "chunk_evidence_count": 0,
                "chunking_skipped": True,
            }
        else:
            if prediction_task_spec.is_pure_binary_root():
                final_prompt = self._build_final_synthesis_prompt(
                    target_condition=target_condition,
                    control_condition=control_condition,
                    chunk_evidence=chunk_evidence,
                    predictor_input=predictor_input,
                    executor_output=executor_output,
                    coverage_summary=coverage_summary,
                    high_priority_context=high_priority_context,
                )
            else:
                final_prompt = self._build_generalized_synthesis_prompt(
                    prediction_task_spec=prediction_task_spec,
                    target_condition=target_condition,
                    control_condition=control_condition,
                    high_priority_context=high_priority_context,
                    chunk_evidence=chunk_evidence,
                    coverage_summary=coverage_summary,
                    non_core_context=executor_output.get("non_core_context_text") or "",
                    chunking_skipped=False,
                )
            predictor_call_context = {
                "mode": "chunked",
                "high_priority_context": high_priority_context,
                "non_core_context": executor_output.get("non_core_context_text") or "",
                "chunk_evidence_count": len(chunk_evidence),
                "chunking_skipped": False,
            }
        prompt_tokens = self._token_count(final_prompt or "")
        predictor_call_context["final_prompt_char_count"] = len(final_prompt or "")
        predictor_call_context["final_prompt_token_estimate"] = prompt_tokens
        if self._prompt_compaction_meta:
            predictor_call_context["prompt_compaction"] = dict(self._prompt_compaction_meta)
        executor_output["predictor_call_context"] = predictor_call_context

        try:
            prediction_data = self._call_predictor_json(
                system_prompt=self.system_prompt,
                user_prompt=final_prompt,
                max_retries=2,
            )
        except Exception as e:
            logger.exception("Predictor failed to obtain JSON response; using deterministic fallback.")
            self._log_error(f"Predictor LLM failure, using fallback prediction: {e}")
            prediction_data = self._build_fallback_prediction_data(
                participant_id=participant_id,
                error=str(e),
                prediction_task_spec=prediction_task_spec,
            )

        if prediction_task_spec.is_pure_binary_root():
            result = self._parse_prediction(
                prediction_data=prediction_data,
                participant_id=participant_id,
                target_condition=target_condition,
                control_condition=control_condition,
                executor_output=executor_output,
                iteration=iteration,
                coverage_summary=coverage_summary,
            )
        else:
            result = self._parse_generalized_prediction(
                prediction_data=prediction_data,
                participant_id=participant_id,
                target_condition=target_condition,
                control_condition=control_condition,
                prediction_task_spec=prediction_task_spec,
                executor_output=executor_output,
                iteration=iteration,
                coverage_summary=coverage_summary,
            )

        if result.binary_classification is not None and result.probability_score is not None:
            self._log_complete(
                f"{result.binary_classification.value} (p={result.probability_score:.3f}, "
                f"confidence={result.confidence_level.value})"
            )
        else:
            self._log_complete(
                f"non-binary output ({prediction_task_spec.root.mode.value}, confidence={result.confidence_level.value})"
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
        backend_is_local = self._is_local_backend()
        last_error: Optional[str] = None
        max_tokens = int(self.LLM_MAX_TOKENS or 4096)
        effective_retries = int(max_retries)
        if backend_is_local:
            effective_retries = max(effective_retries, 4)
        base_prompt = self._fit_prompt_to_input_budget(user_prompt, max_completion_tokens=max_tokens)

        for attempt in range(effective_retries + 1):
            keep_ratio = self._retry_keep_ratio(attempt=attempt, local_backend=backend_is_local)
            prompt = base_prompt
            if keep_ratio < 0.999:
                prompt = self._truncate_prompt_for_retry(base_prompt, keep_ratio=keep_ratio)
            if attempt > 0:
                prompt = (
                    prompt
                    + "\n\nPREVIOUS_ERROR:\n"
                    + str(last_error or "unknown")
                    + "\nReturn only fixed valid JSON. Keep output concise."
                )
            try:
                response = self.llm_client.call(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.LLM_MODEL,
                    max_tokens=max_tokens,
                    temperature=self.LLM_TEMPERATURE,
                    response_format={"type": "json_object"},
                )
                self._record_tokens(response.prompt_tokens, response.completion_tokens)
                try:
                    parsed = parse_json_response(
                        response.content,
                        expected_keys=self._predictor_expected_keys(),
                    )
                    if isinstance(parsed, dict) and parsed:
                        return parsed
                    if isinstance(parsed, dict):
                        raise ValueError("Predictor JSON output was empty.")
                    raise ValueError(f"Predictor JSON output type was {type(parsed).__name__}, expected object.")
                except Exception as parse_exc:
                    if backend_is_local:
                        max_agent_out = int(
                            getattr(self.settings.token_budget, "max_agent_output_tokens", 16000) or 16000
                        )
                        # Local models often need a higher temperature for repair to break loop
                        repaired = self._call_predictor_json_repair(
                            model=self.LLM_MODEL,
                            raw_text=str(response.content or ""),
                            max_tokens=max(2048, min(int(max_tokens), max_agent_out)),
                        )
                        if repaired is not None:
                            return repaired
                    raise parse_exc
            except Exception as e:
                last_error = str(e)
                if self._is_length_error(last_error):
                    max_tokens = max(768, int(max_tokens * 0.75))
                else:
                    max_tokens = max(768, int(max_tokens * 0.9))
                if attempt == effective_retries:
                    raise
        return {}

    def _call_predictor_json_repair(
        self,
        *,
        model: str,
        raw_text: str,
        max_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        if not str(raw_text or "").strip():
            return None
        try:
            repair_resp = self.llm_client.call(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You repair malformed JSON outputs from another model. "
                            "Return ONLY one valid JSON object, no markdown, no explanation."
                        ),
                    },
                    {"role": "user", "content": raw_text},
                ],
                model=model,
                max_tokens=int(max_tokens),
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            self._record_tokens(repair_resp.prompt_tokens, repair_resp.completion_tokens)
            repaired = parse_json_response(
                repair_resp.content,
                expected_keys=self._predictor_expected_keys(),
            )
            if isinstance(repaired, dict) and repaired:
                return repaired
        except Exception as repair_exc:
            logger.warning(
                "[Predictor] JSON repair pass failed: %s: %s",
                type(repair_exc).__name__,
                repair_exc,
            )
        return None

    def _is_local_backend(self) -> bool:
        backend_value = getattr(self.settings.models.backend, "value", self.settings.models.backend)
        return (
            self.settings.models.backend == LLMBackend.LOCAL
            or str(backend_value).lower() == "local"
        )

    def _token_count(self, text: str) -> int:
        raw = str(text or "")
        if not raw:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(raw))
            except Exception:
                pass
        # Fallback heuristic (~4 chars/token)
        return max(1, int(len(raw) / 4))

    def _predictor_input_budget_tokens(self, max_completion_tokens: int) -> int:
        configured = int(getattr(self.settings.token_budget, "max_agent_input_tokens", 0) or 0)
        context_window = int(self.settings.effective_context_window(self.LLM_MODEL))
        reserve = max(1024, min(8192, int(max_completion_tokens) + 1024))
        by_context = max(2048, context_window - reserve)
        if configured > 0:
            return max(2048, min(configured, by_context))
        return by_context

    def _fit_prompt_to_input_budget(self, prompt: str, *, max_completion_tokens: int) -> str:
        budget = self._predictor_input_budget_tokens(max_completion_tokens=max_completion_tokens)
        current = self._token_count(prompt)
        if current <= budget:
            return prompt
        ratio = max(0.25, float(budget) / float(max(1, current)))
        trimmed = self._truncate_prompt_for_retry(prompt, keep_ratio=ratio)
        logger.warning(
            "[Predictor] Prompt exceeded input budget (%s > %s tokens). Truncated with ratio=%.3f.",
            current,
            budget,
            ratio,
        )
        return trimmed

    def _retry_keep_ratio(self, *, attempt: int, local_backend: bool) -> float:
        if attempt <= 0:
            return 1.0
        local_schedule = [0.90, 0.78, 0.66, 0.54, 0.45]
        public_schedule = [0.92, 0.80, 0.68]
        schedule = local_schedule if local_backend else public_schedule
        idx = min(attempt - 1, len(schedule) - 1)
        return float(schedule[idx])

    def _predictor_expected_keys(self) -> List[str]:
        active = self._active_prediction_task_spec
        if active is not None and not active.is_pure_binary_root():
            return [
                "prediction_id",
                "root_prediction",
                "confidence_level",
                "key_findings",
                "reasoning_chain",
                "clinical_summary",
                "uncertainty_factors",
            ]
        return [
            "prediction_id",
            "binary_classification",
            "probability_score",
            "confidence_level",
            "key_findings",
            "reasoning_chain",
            "clinical_summary",
            "supporting_evidence",
            "uncertainty_factors",
        ]

    def _is_length_error(self, error_text: Optional[str]) -> bool:
        text = str(error_text or "").lower()
        return ("finish_reason=length" in text) or ("empty response" in text and "length" in text)

    def _truncate_prompt_for_retry(self, prompt: str, keep_ratio: float) -> str:
        raw_prompt = str(prompt or "")
        if not raw_prompt:
            return ""
        ratio = min(0.95, max(0.2, float(keep_ratio)))

        if self._encoder is not None:
            try:
                tokens = self._encoder.encode(raw_prompt)
                if tokens:
                    keep = max(256, int(len(tokens) * ratio))
                    if keep >= len(tokens):
                        return raw_prompt
                    head = max(128, int(keep * 0.7))
                    tail = max(128, keep - head)
                    if head + tail > len(tokens):
                        head = max(1, len(tokens) // 2)
                        tail = len(tokens) - head
                    front_text = self._encoder.decode(tokens[:head])
                    back_text = self._encoder.decode(tokens[-tail:])
                    return (
                        front_text
                        + "\n\n[TRUNCATED FOR RETRY: middle context omitted due to length constraints]\n\n"
                        + back_text
                    )
            except Exception:
                pass

        keep_chars = max(512, int(len(raw_prompt) * ratio))
        if keep_chars >= len(raw_prompt):
            return raw_prompt
        head_chars = max(256, int(keep_chars * 0.7))
        tail_chars = max(256, keep_chars - head_chars)
        return (
            raw_prompt[:head_chars]
            + "\n\n[TRUNCATED FOR RETRY: middle context omitted due to length constraints]\n\n"
            + raw_prompt[-tail_chars:]
        )

    def _build_fallback_prediction_data(
        self,
        participant_id: str,
        error: str,
        prediction_task_spec: Optional[PredictionTaskSpec] = None,
    ) -> Dict[str, Any]:
        """Deterministic fallback payload when predictor model fails repeatedly."""
        if prediction_task_spec is not None and not prediction_task_spec.is_pure_binary_root():
            return self._build_generalized_fallback_prediction_data(
                participant_id=participant_id,
                error=error,
                prediction_task_spec=prediction_task_spec,
            )
        return {
            "prediction_id": f"fallback_{participant_id}_{str(uuid.uuid4())[:8]}",
            "binary_classification": "CONTROL",
            "probability_score": 0.499,
            "confidence_level": "LOW",
            "key_findings": [
                {
                    "domain": "SYSTEM",
                    "finding": "Predictor LLM response unavailable (length/empty/invalid).",
                    "direction": "NORMAL",
                    "z_score": None,
                    "relevance_to_prediction": "Conservative fallback to avoid unsupported CASE call.",
                }
            ],
            "reasoning_chain": [
                "Model output could not be parsed after retries.",
                "Applied deterministic conservative fallback (CONTROL).",
            ],
            "clinical_summary": (
                "Predictor response was unavailable after retries; conservative fallback was applied. "
                "Re-run with a larger-capability model or lower prompt volume."
            ),
            "supporting_evidence": {
                "for_case": [],
                "for_control": [
                    "Fallback safety policy triggered due to predictor model failure.",
                    f"Raw predictor error: {error}",
                ],
            },
            "uncertainty_factors": [
                "Predictor LLM failure (empty/non-JSON output).",
                "Result generated by deterministic fallback policy.",
            ],
        }

    def _build_final_synthesis_prompt(
        self,
        *,
        target_condition: str,
        control_condition: str,
        chunk_evidence: List[Dict[str, Any]],
        predictor_input: Dict[str, Any],
        executor_output: Dict[str, Any],
        coverage_summary: Dict[str, Any],
        high_priority_context: Optional[str] = None,
    ) -> str:
        if high_priority_context is None:
            high_priority_context = self._build_high_priority_context(predictor_input, executor_output)
        chunks_text = self._render_chunk_evidence_for_prompt(chunk_evidence)
        coverage_text = json_to_toon(self._compact_coverage_summary(coverage_summary))

        return "\n".join(
            [
                "Synthesize final CASE/CONTROL verdict from chunk evidence.",
                f"Target condition: {target_condition}",
                f"Control condition: {control_condition}",
                "Respect calibration and avoid false positives.",
                "Default to CONTROL if class text is ambiguous.",
                "You MUST integrate all chunk evidence rows.",
                "Use professional English for all narrative fields.",
                "Note: processed raw low-priority multimodal data was excluded from chunk evidence to avoid re-processing."
                if executor_output.get("processed_raw_excluded") else "",
                "Return ONLY one valid JSON object. No markdown, no prose, no <think> blocks.",
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
                '  "probability_score": 0.73,',
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

    def _compact_coverage_summary(self, coverage_summary: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(coverage_summary, dict):
            return {}
        compact = dict(coverage_summary)
        missing = compact.get("missing_features")
        if isinstance(missing, list) and len(missing) > 24:
            compact["missing_features_count"] = len(missing)
            compact["missing_features_sample"] = [str(v) for v in missing[:24]]
            compact.pop("missing_features", None)
        return compact

    def _chunk_evidence_token_budget(self) -> int:
        max_input = int(getattr(self.settings.token_budget, "max_agent_input_tokens", 30000) or 30000)
        if self._is_local_backend():
            return max(2500, min(7000, int(max_input * 0.30)))
        return max(8000, min(40000, int(max_input * 0.55)))

    def _trim_text(self, value: Any, max_chars: int) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _compact_chunk_evidence_rows(
        self,
        chunk_evidence: List[Dict[str, Any]],
        *,
        summary_chars: int,
        evidence_items: int,
        evidence_chars: int,
        finding_items: int,
        finding_chars: int,
        feature_key_items: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for row in chunk_evidence:
            if not isinstance(row, dict):
                continue
            key_findings = []
            for finding in row.get("key_findings", [])[:finding_items]:
                if not isinstance(finding, dict):
                    continue
                key_findings.append(
                    {
                        "domain": self._trim_text(finding.get("domain", ""), 48),
                        "finding": self._trim_text(finding.get("finding", ""), finding_chars),
                        "direction": self._trim_text(finding.get("direction", "NORMAL"), 24),
                        "z_score": finding.get("z_score"),
                    }
                )

            row_obj = {
                "chunk": f"{row.get('chunk_index', '?')}/{row.get('chunk_total', '?')}",
                "summary": self._trim_text(row.get("summary", ""), summary_chars),
                "for_case": [self._trim_text(v, evidence_chars) for v in self._as_str_list(row.get("for_case"))[:evidence_items]],
                "for_control": [self._trim_text(v, evidence_chars) for v in self._as_str_list(row.get("for_control"))[:evidence_items]],
                "uncertainty_factors": [
                    self._trim_text(v, evidence_chars)
                    for v in self._as_str_list(row.get("uncertainty_factors"))[:evidence_items]
                ],
                "key_findings": key_findings,
            }
            if feature_key_items > 0:
                row_obj["cited_feature_keys_sample"] = [
                    self._trim_text(v, 120)
                    for v in self._as_str_list(row.get("cited_feature_keys"))[:feature_key_items]
                ]
            rows.append(row_obj)
        return rows

    def _render_chunk_evidence_for_prompt(self, chunk_evidence: List[Dict[str, Any]]) -> str:
        token_budget = self._chunk_evidence_token_budget()
        profiles = [
            {
                "summary_chars": 260,
                "evidence_items": 3,
                "evidence_chars": 140,
                "finding_items": 3,
                "finding_chars": 160,
                "feature_key_items": 6,
                "label": "rich",
            },
            {
                "summary_chars": 180,
                "evidence_items": 2,
                "evidence_chars": 120,
                "finding_items": 2,
                "finding_chars": 120,
                "feature_key_items": 4,
                "label": "balanced",
            },
            {
                "summary_chars": 110,
                "evidence_items": 1,
                "evidence_chars": 90,
                "finding_items": 1,
                "finding_chars": 90,
                "feature_key_items": 0,
                "label": "minimal",
            },
        ]

        last_text = json_to_toon(chunk_evidence)
        selected_label = "raw"
        selected_rows = len(chunk_evidence)
        selected_tokens = self._token_count(last_text)

        for profile in profiles:
            compact_rows = self._compact_chunk_evidence_rows(chunk_evidence, **{k: v for k, v in profile.items() if k != "label"})
            text = json_to_toon(compact_rows)
            tokens = self._token_count(text)
            selected_label = profile["label"]
            selected_rows = len(compact_rows)
            selected_tokens = tokens
            last_text = text
            if tokens <= token_budget:
                break

        self._prompt_compaction_meta = {
            "chunk_rows_in": len(chunk_evidence),
            "chunk_rows_out": selected_rows,
            "chunk_text_tokens": selected_tokens,
            "chunk_token_budget": token_budget,
            "chunk_profile": selected_label,
        }
        return last_text

    def _build_high_priority_context(
        self,
        predictor_input: Dict[str, Any],
        executor_output: Dict[str, Any],
    ) -> str:
        non_num = predictor_input.get("non_numerical_data_raw") or executor_output.get("non_numerical_data") or ""
        dev = predictor_input.get("hierarchical_deviation_raw") or executor_output.get("hierarchical_deviation") or {}
        overview = executor_output.get("data_overview") or {}

        per_section_cap = 900 if self._is_local_backend() else 2500

        def _limit(text: str, max_tokens: Optional[int] = None) -> str:
            raw = str(text or "")
            cap = int(max_tokens or per_section_cap)
            tok_count = self._token_count(raw)
            if tok_count <= cap:
                return raw
            ratio = float(cap) / float(max(1, tok_count))
            return self._truncate_prompt_for_retry(raw, keep_ratio=ratio)

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
        differential_text = _tool_text("DifferentialDiagnosis")

        rows = [
            f"## non_numerical_data_raw\n{_limit(str(non_num))}",
            f"## hierarchical_deviation_raw\n{_limit(json_to_toon(dev))}",
            f"## data_overview\n{_limit(json_to_toon(overview))}",
            f"## phenotype_representation\n{_limit(phenotype_text)}",
            f"## feature_synthesizer\n{_limit(feature_text)}",
            f"## differential_diagnosis\n{_limit(differential_text)}",
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

    def _build_direct_synthesis_prompt(
        self,
        *,
        target_condition: str,
        control_condition: str,
        predictor_input: Dict[str, Any],
        executor_output: Dict[str, Any],
        coverage_summary: Dict[str, Any],
        high_priority_context: Optional[str] = None,
        non_core_context: Optional[str] = None,
    ) -> str:
        if high_priority_context is None:
            high_priority_context = self._build_high_priority_context(predictor_input, executor_output)
        if non_core_context is None:
            non_core_context = executor_output.get("non_core_context_text") or "Not provided"
        coverage_text = json_to_toon(self._compact_coverage_summary(coverage_summary))

        if self._is_local_backend():
            non_core_tokens = self._token_count(str(non_core_context or ""))
            non_core_budget = max(2000, min(8000, int(self._predictor_input_budget_tokens(max_completion_tokens=int(self.LLM_MAX_TOKENS or 4096)) * 0.35)))
            if non_core_tokens > non_core_budget:
                ratio = float(non_core_budget) / float(max(1, non_core_tokens))
                non_core_context = self._truncate_prompt_for_retry(str(non_core_context), keep_ratio=ratio)

        prompt = "\n".join(
            [
                "Synthesize final CASE/CONTROL verdict from full non-core context (no chunk evidence required).",
                f"Target condition: {target_condition}",
                f"Control condition: {control_condition}",
                "Respect calibration and avoid false positives.",
                "Default to CONTROL if class text is ambiguous.",
                "Use professional English for all narrative fields.",
                "Return ONLY one valid JSON object. No markdown, no prose, no <think> blocks.",
                "",
                "## High-priority context (triad)",
                f"```text\n{high_priority_context}\n```",
                "",
                "## Non-core context (direct, unchunked)",
                f"```text\n{non_core_context}\n```",
                "",
                "## Coverage summary",
                f"```text\n{coverage_text}\n```",
                "",
                "Return JSON with fields:",
                "{",
                '  "prediction_id": "string",',
                '  "binary_classification": "CASE|CONTROL|free-text",',
                '  "probability_score": 0.73,',
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
        return self._append_runtime_instruction(prompt, label="Predictor Runtime Instruction")

    def _build_generalized_synthesis_prompt(
        self,
        *,
        prediction_task_spec: PredictionTaskSpec,
        target_condition: str,
        control_condition: str,
        high_priority_context: str,
        chunk_evidence: List[Dict[str, Any]],
        coverage_summary: Dict[str, Any],
        non_core_context: str,
        chunking_skipped: bool,
    ) -> str:
        task_spec_text = (
            prediction_task_spec.model_dump_json(indent=2)
            if hasattr(prediction_task_spec, "model_dump_json")
            else json.dumps(prediction_task_spec.dict(), indent=2)
        )
        coverage_text = json_to_toon(self._compact_coverage_summary(coverage_summary))
        chunk_text = self._render_chunk_evidence_for_prompt(chunk_evidence) if chunk_evidence else "Not provided"
        non_core = str(non_core_context or "Not provided")
        if self._is_local_backend() and len(non_core) > 60000:
            non_core = self._truncate_prompt_for_retry(non_core, keep_ratio=0.4)

        prompt = "\n".join(
            [
                "Synthesize phenotype prediction outputs according to the provided task specification.",
                f"Target label (legacy): {target_condition}",
                f"Control/comparator label (legacy): {control_condition}",
                "Do not assume binary-only output. Follow node modes exactly.",
                "Use professional English for all narrative fields.",
                "Do not use template default values (such as 0.0) unless a true zero estimate is explicitly justified.",
                "Return ONLY one valid JSON object. No markdown, no prose.",
                "",
                "## Prediction task specification (canonical)",
                f"```json\n{task_spec_text}\n```",
                "",
                "## High-priority context",
                f"```text\n{high_priority_context}\n```",
                "",
                "## Coverage summary",
                f"```text\n{coverage_text}\n```",
                "",
                "## Evidence context",
                "Chunking mode: skipped" if chunking_skipped else "Chunking mode: chunked",
                f"```text\n{chunk_text if not chunking_skipped else non_core}\n```",
                "",
                "JSON output contract:",
                "{",
                '  "prediction_id": "string",',
                '  "confidence_level": "HIGH|MEDIUM|LOW",',
                '  "key_findings": [{"domain":"string","finding":"string","direction":"NORMAL","z_score":null,"relevance_to_prediction":"string"}],',
                '  "reasoning_chain": ["string"],',
                '  "clinical_summary": "string",',
                '  "supporting_evidence": {"for_target": ["string"], "against_target": ["string"]},',
                '  "uncertainty_factors": ["string"],',
                '  "root_prediction": {',
                '    "node_id": "root_node_id",',
                '    "mode": "binary_classification|multiclass_classification|univariate_regression|multivariate_regression",',
                '    "classification": {"predicted_label":"string","probabilities":{"label":0.73}},',
                '    "regression": {"values":{"output_name":12.34}},',
                '    "confidence_level": "HIGH|MEDIUM|LOW",',
                '    "confidence_score": 0.73,',
                '    "supporting_evidence_for": ["string"],',
                '    "supporting_evidence_against": ["string"],',
                '    "uncertainty_factors": ["string"],',
                '    "children": []',
                "  }",
                "}",
                "For classification nodes include classification and omit regression.",
                "For regression nodes include regression and omit classification.",
                "Children must mirror task_spec child node_ids and modes.",
            ]
        )
        return self._append_runtime_instruction(prompt, label="Predictor Runtime Instruction")

    def _build_generalized_fallback_prediction_data(
        self,
        *,
        participant_id: str,
        error: str,
        prediction_task_spec: PredictionTaskSpec,
    ) -> Dict[str, Any]:
        def _build_fallback_node(node: PredictionTaskNode) -> Dict[str, Any]:
            if node.mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION):
                labels = list(node.class_labels)
                probs = {}
                if labels:
                    uniform = 1.0 / float(len(labels))
                    probs = {label: uniform for label in labels}
                classification = {
                    "predicted_label": labels[0] if labels else "UNKNOWN",
                    "probabilities": probs,
                }
                regression = None
            else:
                outputs = list(node.regression_outputs)
                regression = {"values": {name: 0.0 for name in outputs}}
                classification = None
            return {
                "node_id": node.node_id,
                "mode": node.mode.value,
                "classification": classification,
                "regression": regression,
                "confidence_level": "LOW",
                "confidence_score": 0.2,
                "supporting_evidence_for": [],
                "supporting_evidence_against": [
                    "Fallback safety policy triggered due to predictor model failure."
                ],
                "uncertainty_factors": [f"predictor_fallback_error:{error}"],
                "children": [_build_fallback_node(child) for child in node.children],
            }

        root_node = _build_fallback_node(prediction_task_spec.root)
        return {
            "prediction_id": f"fallback_{participant_id}_{str(uuid.uuid4())[:8]}",
            "confidence_level": "LOW",
            "key_findings": [
                {
                    "domain": "SYSTEM",
                    "finding": "Predictor LLM response unavailable; deterministic task-shaped fallback applied.",
                    "direction": "NORMAL",
                    "z_score": None,
                    "relevance_to_prediction": "Conservative fallback preserving output schema.",
                }
            ],
            "reasoning_chain": [
                "Model output could not be parsed after retries.",
                "Applied deterministic fallback matching requested prediction task schema.",
            ],
            "clinical_summary": (
                "Predictor response was unavailable after retries; deterministic fallback was applied. "
                "Re-run with a higher-capability model or reduced context size."
            ),
            "supporting_evidence": {
                "for_target": [],
                "against_target": [
                    "Predictor model failure; inference reliability reduced.",
                    f"Raw predictor error: {error}",
                ],
            },
            "uncertainty_factors": [
                "Predictor LLM failure (empty/non-JSON output).",
                "Result generated by deterministic fallback policy.",
            ],
            "root_prediction": root_node,
        }

    def _parse_generalized_prediction(
        self,
        *,
        prediction_data: Dict[str, Any],
        participant_id: str,
        target_condition: str,
        control_condition: str,
        prediction_task_spec: PredictionTaskSpec,
        executor_output: Dict[str, Any],
        iteration: int,
        coverage_summary: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        root_payload = prediction_data.get("root_prediction") or {}
        if not isinstance(root_payload, dict):
            raise ValueError("Generalized predictor output missing root_prediction object")

        root_prediction = self._parse_node_prediction(
            node_payload=root_payload,
            node_spec=prediction_task_spec.root,
            path=prediction_task_spec.root.node_id,
        )
        repaired_value_note = self._repair_univariate_zero_from_narrative(
            root_prediction=root_prediction,
            prediction_data=prediction_data,
        )

        confidence = self._parse_confidence_level(
            prediction_data.get("confidence_level") or root_payload.get("confidence_level")
        )

        key_findings: List[KeyFinding] = []
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
        if repaired_value_note:
            uncertainty_factors.append(repaired_value_note)
        if coverage_summary and not coverage_summary.get("invariant_ok", True):
            uncertainty_factors.append(
                f"Coverage invariant warning: {coverage_summary.get('missing_feature_count', 0)} features not represented in evidence."
            )
        uncertainty_factors = self._dedupe_preserve(uncertainty_factors)
        supporting_evidence = prediction_data.get("supporting_evidence", {})
        if not isinstance(supporting_evidence, dict):
            supporting_evidence = {}
        supporting_evidence.setdefault("for_target", [])
        supporting_evidence.setdefault("against_target", [])

        return PredictionResult(
            prediction_id=str(prediction_data.get("prediction_id", str(uuid.uuid4())[:8])),
            participant_id=participant_id,
            target_condition=target_condition,
            control_condition=control_condition,
            prediction_task_spec=prediction_task_spec,
            root_prediction=root_prediction,
            confidence_level=confidence,
            key_findings=key_findings,
            reasoning_chain=reasoning_chain,
            supporting_evidence=supporting_evidence,
            uncertainty_factors=uncertainty_factors,
            clinical_summary=str(prediction_data.get("clinical_summary", "")),
            domains_processed=executor_output.get("domains_processed", []),
            total_tokens_used=executor_output.get("total_tokens_used", 0),
            iteration=iteration,
        )

    def _parse_node_prediction(
        self,
        *,
        node_payload: Dict[str, Any],
        node_spec: PredictionTaskNode,
        path: str,
    ) -> NodePrediction:
        raw_children = node_payload.get("children") or []
        if not isinstance(raw_children, list):
            raw_children = []
        payload_children_by_id = {}
        for child_payload in raw_children:
            if not isinstance(child_payload, dict):
                continue
            cid = str(child_payload.get("node_id") or "").strip()
            if cid:
                payload_children_by_id[cid] = child_payload

        children: List[NodePrediction] = []
        for child_spec in node_spec.children:
            child_payload = payload_children_by_id.get(child_spec.node_id)
            if child_payload is None and not bool(getattr(child_spec, "required", True)):
                continue
            if child_payload is None:
                child_payload = {"node_id": child_spec.node_id}
            child_pred = self._parse_node_prediction(
                node_payload=child_payload,
                node_spec=child_spec,
                path=f"{path}.{child_spec.node_id}",
            )
            children.append(child_pred)

        confidence_level = self._parse_confidence_level(node_payload.get("confidence_level"))
        confidence_score = node_payload.get("confidence_score", 0.5)
        try:
            confidence_score = max(0.0, min(1.0, float(confidence_score)))
        except Exception:
            confidence_score = 0.5

        classification = None
        regression = None
        if node_spec.mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION):
            cls_payload = node_payload.get("classification") or {}
            if not isinstance(cls_payload, dict):
                cls_payload = {}
            probs = cls_payload.get("probabilities") or {}
            if not isinstance(probs, dict):
                probs = {}

            normalized_probs: Dict[str, float] = {}
            for label in node_spec.class_labels:
                raw = probs.get(label)
                if raw is None:
                    continue
                try:
                    normalized_probs[label] = max(0.0, min(1.0, float(raw)))
                except Exception:
                    continue

            if not normalized_probs and node_spec.class_labels:
                uniform = 1.0 / float(len(node_spec.class_labels))
                normalized_probs = {label: uniform for label in node_spec.class_labels}
            elif normalized_probs:
                total = sum(normalized_probs.values())
                if total > 0:
                    normalized_probs = {k: float(v / total) for k, v in normalized_probs.items()}

            predicted_label = str(cls_payload.get("predicted_label") or "").strip()
            if predicted_label not in node_spec.class_labels and normalized_probs:
                predicted_label = max(normalized_probs.items(), key=lambda x: x[1])[0]
            if predicted_label not in node_spec.class_labels and node_spec.class_labels:
                predicted_label = node_spec.class_labels[0]

            classification = ClassificationPrediction(
                predicted_label=predicted_label or "UNKNOWN",
                probabilities=normalized_probs,
            )
        else:
            reg_payload = node_payload.get("regression") or {}
            if not isinstance(reg_payload, dict):
                reg_payload = {}
            values_payload = reg_payload.get("values") or {}
            if not isinstance(values_payload, dict):
                values_payload = {}
            normalized_payload_values: Dict[str, Any] = {}
            payload_numeric_values: List[float] = []
            for raw_key, raw_value in values_payload.items():
                key_norm = self._normalize_output_key(raw_key)
                if key_norm and key_norm not in normalized_payload_values:
                    normalized_payload_values[key_norm] = raw_value
                try:
                    payload_numeric_values.append(float(raw_value))
                except Exception:
                    continue
            values: Dict[str, float] = {}
            for output_name in node_spec.regression_outputs:
                raw = values_payload.get(output_name)
                if raw is None:
                    output_norm = self._normalize_output_key(output_name)
                    if output_norm and output_norm in normalized_payload_values:
                        raw = normalized_payload_values[output_norm]
                if raw is None and len(node_spec.regression_outputs) == 1 and len(payload_numeric_values) == 1:
                    # Univariate recovery path: accept a single numeric value even if key naming differs.
                    raw = payload_numeric_values[0]
                if raw is None:
                    raise ValueError(
                        f"Missing regression output '{output_name}' for node '{node_spec.node_id}'"
                    )
                try:
                    values[output_name] = float(raw)
                except Exception:
                    raise ValueError(
                        f"Non-numeric regression output '{output_name}' for node '{node_spec.node_id}'"
                    )
            regression = RegressionPrediction(values=values)

        return NodePrediction(
            node_id=node_spec.node_id,
            path=path,
            mode=node_spec.mode,
            classification=classification,
            regression=regression,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            supporting_evidence_for=self._as_str_list(node_payload.get("supporting_evidence_for")),
            supporting_evidence_against=self._as_str_list(node_payload.get("supporting_evidence_against")),
            uncertainty_factors=self._as_str_list(node_payload.get("uncertainty_factors")),
            children=children,
        )

    def _parse_prediction(
        self,
        prediction_data: Dict[str, Any],
        participant_id: str,
        target_condition: str,
        control_condition: str,
        executor_output: Dict[str, Any],
        iteration: int,
        coverage_summary: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        raw_probability = self._parse_probability(prediction_data.get("probability_score", 0.49))
        classification, ambiguous_class = self._normalize_classification(
            prediction_data.get("binary_classification"),
            raw_probability,
            control_condition,
        )
        probability = self._normalize_probability_for_classification(raw_probability, classification)

        confidence = self._parse_confidence_level(prediction_data.get("confidence_level", "MEDIUM"))

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

        root_task_spec = build_binary_task_spec(
            target_label=target_condition,
            control_label=control_condition,
        )
        target_label, control_label = root_task_spec.legacy_target_control()
        target_prob = float(max(0.0, min(1.0, probability)))
        control_prob = float(max(0.0, min(1.0, 1.0 - target_prob)))
        predicted_label = target_label if classification == BinaryClassification.CASE else control_label
        root_prediction = NodePrediction(
            node_id=root_task_spec.root.node_id,
            path=root_task_spec.root.node_id,
            mode=PredictionMode.BINARY_CLASSIFICATION,
            classification=ClassificationPrediction(
                predicted_label=predicted_label,
                probabilities={
                    target_label: target_prob,
                    control_label: control_prob,
                },
            ),
            confidence_level=confidence,
            confidence_score=float(max(target_prob, control_prob)),
        )

        return PredictionResult(
            prediction_id=str(prediction_data.get("prediction_id", str(uuid.uuid4())[:8])),
            participant_id=participant_id,
            target_condition=target_condition,
            control_condition=control_condition,
            prediction_task_spec=root_task_spec,
            root_prediction=root_prediction,
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
        control_condition: str,
    ) -> Tuple[BinaryClassification, bool]:
        text = str(raw_classification or "").upper()
        has_case = "CASE" in text or "TARGET PHENOTYPE" in text or "LIKELY HAS TARGET" in text
        control_upper = str(control_condition or "").upper()
        has_control = (
            "CONTROL" in text
            or (control_upper and control_upper in text)
            or "NOT PSYCHIATRIC" in text
            or "NON-PSYCHIATRIC" in text
        )

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

    def _parse_confidence_level(self, value: Any) -> ConfidenceLevel:
        try:
            return ConfidenceLevel(str(value or "MEDIUM").upper())
        except Exception:
            return ConfidenceLevel.MEDIUM

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

    def _normalize_output_key(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        # Normalize punctuation/spacing for robust matching between task outputs and LLM keys.
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    def _repair_univariate_zero_from_narrative(
        self,
        *,
        root_prediction: NodePrediction,
        prediction_data: Dict[str, Any],
    ) -> Optional[str]:
        """
        Recover a likely univariate value when models emit template defaults (0.0),
        using explicit narrative numeric anchors (for example, '82 years/ans').
        """
        if root_prediction.mode != PredictionMode.UNIVARIATE_REGRESSION:
            return None
        reg = root_prediction.regression
        if reg is None or not isinstance(reg.values, dict) or len(reg.values) != 1:
            return None
        output_name, raw_value = next(iter(reg.values.items()))
        try:
            value = float(raw_value)
        except Exception:
            return None
        if abs(value) > 1e-12:
            return None

        summary_text = str(prediction_data.get("clinical_summary") or "")
        reasoning_text = " ".join(
            [str(step or "") for step in list(prediction_data.get("reasoning_chain") or [])]
        )
        text = f"{summary_text} {reasoning_text}".strip()
        if not text:
            return None

        label_norm = self._normalize_output_key(output_name)
        inferred: Optional[float] = None
        if any(token in label_norm for token in ("age", "year", "years", "ans", "longevity", "mortality", "survival")):
            inferred = self._extract_age_like_value(text)
        elif "iq" in label_norm:
            inferred = self._extract_keyword_numeric_value(
                text=text,
                keywords=("iq", "intelligence quotient"),
                min_value=40.0,
                max_value=200.0,
            )

        if inferred is None:
            inferred = self._extract_keyword_numeric_value(
                text=text,
                keywords=(str(output_name),),
                min_value=-1_000_000.0,
                max_value=1_000_000.0,
            )
        if inferred is None or abs(float(inferred)) <= 1e-12:
            return None

        reg.values[output_name] = float(inferred)
        return (
            f"Recovered '{output_name}' from narrative numeric context after template-like zero output."
        )

    def _extract_age_like_value(self, text: str) -> Optional[float]:
        patterns = [
            r"(\d{1,3}(?:\.\d+)?)\s*(?:years?|yrs?|ans|años|jaar)\b",
            r"(?:age|aged|âge|edad)\s*(?:of|de|:)?\s*(\d{1,3}(?:\.\d+)?)\b",
            r"(?:die|death|deces|d[eé]c[eè]s|mortality)\D{0,24}(\d{1,3}(?:\.\d+)?)",
        ]
        candidates: List[float] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                try:
                    value = float(match.group(1))
                except Exception:
                    continue
                if 1.0 <= value <= 130.0:
                    candidates.append(value)
        if not candidates:
            return None
        # Use the last explicit age mention to align with final summary statements.
        return float(candidates[-1])

    def _extract_keyword_numeric_value(
        self,
        *,
        text: str,
        keywords: Tuple[str, ...],
        min_value: float,
        max_value: float,
    ) -> Optional[float]:
        for keyword in keywords:
            kw = str(keyword or "").strip()
            if not kw:
                continue
            pattern = re.escape(kw) + r".{0,24}?([-+]?\d{1,6}(?:\.\d+)?)"
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                try:
                    value = float(match.group(1))
                except Exception:
                    continue
                if min_value <= value <= max_value:
                    return float(value)
        return None

    def _print_prediction_summary(self, result: PredictionResult):
        """Print formatted prediction summary."""
        print(f"\n{'='*60}")
        print("PREDICTION RESULT")
        print(f"{'='*60}")
        if result.binary_classification is not None and result.probability_score is not None:
            print(f"Classification: {result.binary_classification.value}")
            print(f"Probability: {result.probability_score:.1%}")
        else:
            mode = (
                result.prediction_task_spec.root.mode.value
                if result.prediction_task_spec is not None
                else "unknown"
            )
            print(f"Primary mode: {mode}")
            if result.root_prediction is not None:
                if result.root_prediction.classification is not None:
                    print(
                        f"Primary label: {result.root_prediction.classification.predicted_label}"
                    )
                if result.root_prediction.regression is not None:
                    print(f"Primary outputs: {result.root_prediction.regression.values}")
        print(f"Confidence: {result.confidence_level.value}")
        print("\nKey Findings:")
        for i, finding in enumerate(result.key_findings[:5], 1):
            print(f"  {i}. [{finding.domain}] {finding.finding[:60]}...")
        print("\nClinical Summary:")
        print(f"  {result.clinical_summary[:200]}...")
        print(f"{'='*60}\n")
