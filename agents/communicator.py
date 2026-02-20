"""
COMPASS Communicator Agent

Creates a deep, clinician-grade phenotyping report (Markdown).
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tiktoken

from .base_agent import BaseAgent
from ..config.settings import get_settings
from ..tools import get_tool
from ..utils.core.embedding_store import get_embedding_store
from ..utils.toon import json_to_toon
from ..utils.token_packer import count_tokens, truncate_text_by_tokens

logger = logging.getLogger("compass.communicator")


@dataclass
class _CommSection:
    name: str
    text: str
    chunkable: bool = True
    optional: bool = False
    low_priority: bool = False


class Communicator(BaseAgent):
    """
    The Communicator creates a deep phenotyping report for clinicians/researchers.
    """

    AGENT_NAME = "Communicator"
    PROMPT_FILE = "communicator_prompt.txt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings = get_settings()
        self.embedding_store = get_embedding_store()
        self.last_run_metadata: Dict[str, Any] = {}

        # Configure LLM params
        self.LLM_MODEL = self.settings.models.communicator_model
        self.LLM_MAX_TOKENS = self.settings.models.communicator_max_tokens
        self.LLM_TEMPERATURE = self.settings.models.communicator_temperature
        try:
            self._encoder = tiktoken.encoding_for_model(self.LLM_MODEL or "gpt-5")
        except Exception:
            try:
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoder = None

    def execute(
        self,
        prediction: Any,
        evaluation: Any,
        executor_output: Dict[str, Any],
        data_overview: Dict[str, Any],
        execution_summary: Dict[str, Any],
        report_context_note: str = "",
        control_condition: Optional[str] = None,
        user_focus_modalities: str = "",
        user_general_instruction: str = "",
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Generate the deep phenotyping report in Markdown.
        """
        self._log_start("deep phenotyping report generation")

        sections, anchors = self._build_sections(
            prediction=prediction,
            evaluation=evaluation,
            executor_output=executor_output,
            data_overview=data_overview,
            execution_summary=execution_summary,
            report_context_note=report_context_note,
            control_condition=control_condition,
            user_focus_modalities=user_focus_modalities,
            user_general_instruction=user_general_instruction,
        )
        threshold = self._communicator_input_threshold()
        model_hint = self.LLM_MODEL or "gpt-5"
        base_prompt_text = self._build_prompt_header()
        anchors_text = self._render_anchors(anchors)
        base_tokens = count_tokens(base_prompt_text + "\n\n" + anchors_text, model_hint=model_hint)
        available_for_evidence = max(6000, threshold - base_tokens)

        primary_sections = [s for s in sections if not s.optional and not s.low_priority]
        optional_sections = [s for s in sections if s.optional and not s.low_priority]
        low_priority_sections = [s for s in sections if s.low_priority]

        primary_text = self._render_sections(primary_sections)
        primary_tokens = count_tokens(primary_text, model_hint=model_hint)

        chunking_used = False
        chunk_count = 0
        chunk_evidence: List[Dict[str, Any]] = []
        evidence_text = ""

        if primary_tokens <= available_for_evidence:
            evidence_text, consumed = self._fit_sections(
                primary_sections,
                token_budget=available_for_evidence,
                model_hint=model_hint,
            )
            remaining = max(0, available_for_evidence - consumed)
            if remaining > 1200:
                add_text, add_tokens = self._fit_sections(
                    optional_sections + low_priority_sections,
                    token_budget=remaining,
                    model_hint=model_hint,
                )
                if add_text:
                    evidence_text = evidence_text + "\n\n" + add_text
                    remaining = max(0, remaining - add_tokens)
            chunking_reason = "primary_fit_threshold"
        else:
            chunking_used = True
            chunking_reason = "primary_over_threshold"
            chunk_budget = max(12000, min(42000, int(available_for_evidence * 0.55)))
            chunks = self._build_chunks(primary_sections, chunk_budget=chunk_budget, model_hint=model_hint)
            chunk_count = len(chunks)
            if status_callback:
                status_callback(f"Deep phenotype evidence extraction ({chunk_count} chunks)")
            chunk_evidence = self._extract_chunk_evidence(
                chunks=chunks,
                anchors=anchors,
                status_callback=status_callback,
            )
            evidence_text = self._render_chunk_evidence(chunk_evidence)
            remaining = max(
                0,
                available_for_evidence - count_tokens(evidence_text, model_hint=model_hint),
            )
            if remaining > 1200:
                add_text, _ = self._fit_sections(
                    optional_sections + low_priority_sections,
                    token_budget=remaining,
                    model_hint=model_hint,
                )
                if add_text:
                    evidence_text = evidence_text + "\n\n" + add_text

        rag_enabled = bool(str(user_focus_modalities or "").strip() or str(user_general_instruction or "").strip())
        rag_rows: List[Dict[str, Any]] = []
        if rag_enabled:
            prompt_probe = self._compose_prompt(
                base_header=base_prompt_text,
                anchors_text=anchors_text,
                evidence_text=evidence_text,
                rag_text="",
            )
            used = count_tokens(prompt_probe, model_hint=model_hint)
            rag_budget = max(0, threshold - used)
            if rag_budget > 1200:
                rag_rows = self._build_guided_rag_rows(
                    executor_output=executor_output,
                    anchors=anchors,
                    token_budget=rag_budget,
                )
        rag_text = ""
        if rag_rows:
            rag_text = "### Guided Context Expansion\n```text\n" + json_to_toon(rag_rows[:200]) + "\n```"

        user_prompt = self._compose_prompt(
            base_header=base_prompt_text,
            anchors_text=anchors_text,
            evidence_text=evidence_text,
            rag_text=rag_text,
        )
        final_tokens = count_tokens(user_prompt, model_hint=model_hint)
        if final_tokens > threshold:
            user_prompt = truncate_text_by_tokens(user_prompt, threshold, model_hint=model_hint, suffix="")
            final_tokens = threshold
        max_completion_tokens = int(self.LLM_MAX_TOKENS or self.settings.models.tool_max_tokens or 16000)
        fallback_used = False
        fallback_reason = ""
        fallback_prompt_tokens = 0
        fallback_max_completion_tokens = 0
        try:
            response = self._invoke_llm_with_limits(
                user_prompt,
                max_tokens=max_completion_tokens,
                max_retries=0,
            )
        except Exception as exc:
            if not self._is_length_exhaustion_error(exc):
                raise

            fallback_used = True
            fallback_reason = str(exc)
            fallback_threshold = threshold
            fallback_available = max(10000, fallback_threshold - base_tokens)

            # Preserve high-information coverage by compressing primary sections with chunk evidence
            # before reducing prompt size.
            compact_primary_text = ""
            compact_chunk_count = 0
            if primary_sections:
                fallback_chunk_budget = max(12000, min(36000, int(fallback_available * 0.45)))
                fallback_chunks = self._build_chunks(
                    primary_sections,
                    chunk_budget=fallback_chunk_budget,
                    model_hint=model_hint,
                )
                compact_chunk_count = len(fallback_chunks)
                if len(fallback_chunks) > 1:
                    fallback_chunk_rows = self._extract_chunk_evidence(
                        chunks=fallback_chunks,
                        anchors=anchors,
                        status_callback=None,
                    )
                    compact_primary_text = self._render_chunk_evidence(fallback_chunk_rows)

            if not compact_primary_text:
                compact_primary_text, _ = self._fit_sections(
                    primary_sections,
                    token_budget=fallback_available,
                    model_hint=model_hint,
                )

            # Refill remaining space with optional/low-priority context.
            compact_consumed = count_tokens(compact_primary_text, model_hint=model_hint)
            compact_remaining = max(0, fallback_available - compact_consumed)
            if compact_remaining > 1200:
                compact_add_text, _ = self._fit_sections(
                    optional_sections + low_priority_sections,
                    token_budget=compact_remaining,
                    model_hint=model_hint,
                )
                if compact_add_text:
                    compact_primary_text = compact_primary_text + "\n\n" + compact_add_text

            strict_header = (
                base_prompt_text
                + "\n\n### Output Constraints\n"
                + "- Start writing the final report immediately.\n"
                + "- Keep total output concise and structured; avoid redundant repetition.\n"
                + "- Prioritize tables and domain summaries over long prose.\n"
            )
            compact_prompt = self._compose_prompt(
                base_header=strict_header,
                anchors_text=anchors_text,
                evidence_text=compact_primary_text,
                rag_text=rag_text if rag_enabled else "",
            )
            fallback_prompt_tokens = count_tokens(compact_prompt, model_hint=model_hint)
            if fallback_prompt_tokens > fallback_threshold:
                compact_prompt = truncate_text_by_tokens(
                    compact_prompt,
                    fallback_threshold,
                    model_hint=model_hint,
                    suffix="",
                )
                fallback_prompt_tokens = fallback_threshold
            fallback_max_completion_tokens = min(max_completion_tokens, 24000)
            response = self._invoke_llm_with_limits(
                compact_prompt,
                max_tokens=fallback_max_completion_tokens,
                max_retries=0,
            )
            chunk_count = max(chunk_count, compact_chunk_count)
            chunking_used = chunking_used or compact_chunk_count > 1
            if compact_chunk_count > 1:
                chunking_reason = "length_recovery_chunk_compress"
        content = response.get("content", "").strip()
        content = self._normalize_markdown_tables(content)
        self.last_run_metadata = {
            "input_threshold_tokens": threshold,
            "base_tokens": base_tokens,
            "primary_tokens": primary_tokens,
            "final_prompt_tokens": final_tokens,
            "chunking_used": chunking_used,
            "chunking_reason": chunking_reason,
            "chunk_count": chunk_count,
            "chunk_evidence_count": len(chunk_evidence),
            "rag_enabled": rag_enabled,
            "rag_added_count": len(rag_rows),
            "user_focus_modalities_present": bool(str(user_focus_modalities or "").strip()),
            "user_general_instruction_present": bool(str(user_general_instruction or "").strip()),
            "embedding_store_path": str(getattr(self.embedding_store, "db_path", "unknown")),
            "embedding_store_fallback": bool(getattr(self.embedding_store, "fallback_reason", None)),
            "embedding_store_fallback_reason": getattr(self.embedding_store, "fallback_reason", None),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "fallback_prompt_tokens": fallback_prompt_tokens,
            "fallback_max_completion_tokens": fallback_max_completion_tokens,
        }

        self._log_complete("deep_phenotype.md created")
        return content

    def execute_xai_report(
        self,
        *,
        xai_result: Dict[str, Any],
        prediction: Any,
        evaluation: Any,
        execution_summary: Dict[str, Any],
        target_condition: str,
        control_condition: str,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Generate a structured explainability report in Markdown from XAI outputs.
        """
        self._log_start("xai explainability report generation")
        model_hint = self.LLM_MODEL or "gpt-5"
        threshold = self._communicator_input_threshold()
        max_completion_tokens = int(self.LLM_MAX_TOKENS or self.settings.models.tool_max_tokens or 16000)

        prompt = self._build_xai_prompt(
            xai_result=xai_result,
            prediction=prediction,
            evaluation=evaluation,
            execution_summary=execution_summary,
            target_condition=target_condition,
            control_condition=control_condition,
        )
        prompt_tokens = count_tokens(prompt, model_hint=model_hint)
        if prompt_tokens > threshold:
            prompt = truncate_text_by_tokens(prompt, threshold, model_hint=model_hint, suffix="")
            prompt_tokens = threshold

        eta_seconds = max(12, int((prompt_tokens + max_completion_tokens) / 220))
        if status_callback:
            status_callback(f"Generating explainability report (~{eta_seconds}s ETA)")

        started = time.time()
        fallback_used = False
        fallback_reason = ""
        fallback_prompt_tokens = 0
        fallback_max_completion_tokens = 0
        try:
            response = self._invoke_llm_with_limits(
                prompt,
                max_tokens=max_completion_tokens,
                max_retries=0,
            )
        except Exception as exc:
            if not self._is_length_exhaustion_error(exc):
                raise
            fallback_used = True
            fallback_reason = str(exc)
            compact_prompt = truncate_text_by_tokens(
                prompt,
                int(max(4000, threshold * 0.75)),
                model_hint=model_hint,
                suffix="",
            )
            fallback_prompt_tokens = count_tokens(compact_prompt, model_hint=model_hint)
            fallback_max_completion_tokens = min(max_completion_tokens, 20000)
            response = self._invoke_llm_with_limits(
                compact_prompt,
                max_tokens=fallback_max_completion_tokens,
                max_retries=0,
            )

        content = self._normalize_markdown_tables((response.get("content") or "").strip())
        duration = round(time.time() - started, 3)
        self.last_run_metadata = {
            "report_type": "xai",
            "input_threshold_tokens": threshold,
            "final_prompt_tokens": prompt_tokens,
            "max_completion_tokens": max_completion_tokens,
            "eta_seconds_estimate": eta_seconds,
            "duration_seconds": duration,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "fallback_prompt_tokens": fallback_prompt_tokens,
            "fallback_max_completion_tokens": fallback_max_completion_tokens,
            "methods_requested": list((xai_result or {}).get("methods_requested") or []),
            "methods_in_report": sorted(list(((xai_result or {}).get("methods") or {}).keys())),
        }
        self._log_complete("xai_explainability_report.md created")
        return content

    @staticmethod
    def _normalize_markdown_tables(text: str) -> str:
        lines = text.splitlines()
        fixed: List[str] = []

        for line in lines:
            if "Table:" in line and "|" in line and line.count("|") >= 3:
                title, rest = line.split("|", 1)
                title = title.strip()
                rest = "|" + rest.lstrip()
                rest = rest.replace(" |", "\n|")
                rest = re.sub(r"(?<!\n)\|---", "\n|---", rest)
                fixed.append(title)
                fixed.append("")
                fixed.append(rest)
                continue
            fixed.append(line)

        return "\n".join(fixed)

    @staticmethod
    def _is_length_exhaustion_error(exc: Exception) -> bool:
        message = str(exc or "").lower()
        return (
            "finish_reason=length" in message
            or "returned empty response" in message
            or "max_tokens" in message
            or "max completion" in message
        )

    def _invoke_llm_with_limits(self, user_prompt: str, *, max_tokens: int, max_retries: int) -> Dict[str, Any]:
        """
        Call `_call_llm` with explicit limits, while remaining compatible with tests
        that monkeypatch `_call_llm` using a simplified signature.
        """
        try:
            return self._call_llm(
                user_prompt,
                expect_json=False,
                max_retries=max_retries,
                max_tokens=max_tokens,
            )
        except TypeError as exc:
            if "unexpected keyword argument" in str(exc).lower():
                return self._call_llm(user_prompt, expect_json=False)
            raise

    def _build_xai_prompt(
        self,
        *,
        xai_result: Dict[str, Any],
        prediction: Any,
        evaluation: Any,
        execution_summary: Dict[str, Any],
        target_condition: str,
        control_condition: str,
    ) -> str:
        methods = dict((xai_result or {}).get("methods") or {})
        method_blocks: List[Dict[str, Any]] = []
        for name in sorted(methods.keys()):
            info = methods.get(name) or {}
            block = {
                "method": name,
                "status": info.get("status"),
                "model": info.get("model"),
                "duration_seconds": info.get("duration_seconds"),
                "parameters": {
                    "k": info.get("k"),
                    "runs": info.get("runs"),
                    "adaptive": info.get("adaptive"),
                    "steps": info.get("steps"),
                    "baseline_mode": info.get("baseline_mode"),
                    "span_mode": info.get("span_mode"),
                    "repeats": info.get("repeats"),
                    "temperature": info.get("temperature"),
                },
                "top_leaf_features": (info.get("top_leaf_features") or [])[:25],
                "top_parent_scores_l1": sorted(
                    (info.get("parent_scores_l1") or {}).items(),
                    key=lambda x: float(x[1]),
                    reverse=True,
                )[:15],
                "error": info.get("error"),
            }
            method_blocks.append(block)

        prediction_text = json_to_toon(self._to_dict(prediction))
        evaluation_text = json_to_toon(self._to_dict(evaluation))
        summary_text = json_to_toon(execution_summary or {})
        xai_text = json_to_toon(
            {
                "enabled": (xai_result or {}).get("enabled"),
                "status": (xai_result or {}).get("status"),
                "methods_requested": (xai_result or {}).get("methods_requested"),
                "feature_space": (xai_result or {}).get("feature_space"),
                "methods": method_blocks,
            }
        )

        return "\n".join(
            [
                "You are the COMPASS Communicator Agent.",
                "Produce a comprehensive explainability report in Markdown for deep phenotype prediction outputs.",
                "",
                "CRITICAL RULES:",
                "1) Use only provided data. Do not hallucinate.",
                "2) If a value is absent, state 'not provided'.",
                "3) Keep method outputs traceable and distinguish method-specific findings from consensus.",
                "4) Do not claim causal validity; report attribution evidence and limitations explicitly.",
                "5) All tables must be valid Markdown tables.",
                "",
                "REQUIRED OUTPUT STRUCTURE:",
                "- Title and context",
                "- Prediction snapshot",
                "- Method configuration and runtime summary",
                "- Feature importance results by method",
                "- Cross-method agreement and disagreement",
                "- Reliability, caveats, and dataflow checks",
                "- Appendix with top features and traceability snippets",
                "",
                f"Target condition: {target_condition}",
                f"Control condition: {control_condition}",
                "",
                "## Prediction",
                f"```text\n{prediction_text}\n```",
                "",
                "## Critic Evaluation",
                f"```text\n{evaluation_text}\n```",
                "",
                "## Execution Summary",
                f"```text\n{summary_text}\n```",
                "",
                "## XAI Inputs",
                f"```text\n{xai_text}\n```",
                "",
                "Now produce the final Markdown report only.",
            ]
        )

    def _effective_context_limit(self) -> int:
        return int(self.settings.effective_context_window(self.LLM_MODEL))

    def _communicator_input_threshold(self) -> int:
        return int(0.9 * self._effective_context_limit())

    @staticmethod
    def _to_dict(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return obj

    def _tool_group_text(self, step_outputs: Dict[Any, Any], tool_names: Sequence[str]) -> str:
        allowed = {str(t) for t in tool_names}
        rows: List[str] = []
        for sid, out in sorted(step_outputs.items(), key=lambda x: str(x[0])):
            if not isinstance(out, dict):
                continue
            name = out.get("tool_name") or (out.get("_step_meta") or {}).get("tool_name")
            if name not in allowed:
                continue
            rows.append(f"## Step {sid} ({name})\n{json_to_toon(out)}")
        return "\n\n".join(rows) if rows else "Not provided"

    def _build_sections(
        self,
        *,
        prediction: Any,
        evaluation: Any,
        executor_output: Dict[str, Any],
        data_overview: Dict[str, Any],
        execution_summary: Dict[str, Any],
        report_context_note: str,
        control_condition: Optional[str],
        user_focus_modalities: str,
        user_general_instruction: str,
    ) -> Tuple[List[_CommSection], Dict[str, str]]:
        prediction_dict = self._to_dict(prediction)
        evaluation_dict = self._to_dict(evaluation)
        predictor_input = executor_output.get("predictor_input", {}) or {}
        fusion_result = self._to_dict(executor_output.get("fusion_result"))
        step_outputs = executor_output.get("step_outputs", {}) or {}

        non_num = predictor_input.get("non_numerical_data_raw") or executor_output.get("non_numerical_data") or ""
        deviation = predictor_input.get("hierarchical_deviation_raw") or executor_output.get("hierarchical_deviation") or {}
        control = (
            control_condition
            or prediction_dict.get("control_condition")
            or executor_output.get("control_condition")
            or "Control condition not provided"
        )

        sections: List[_CommSection] = [
            _CommSection("non_numerical_raw", json_to_toon(non_num), chunkable=True),
            _CommSection("hierarchical_deviation_raw", json_to_toon(deviation), chunkable=True),
            _CommSection("data_overview", json_to_toon(data_overview), chunkable=True),
            _CommSection("phenotype_representation", self._tool_group_text(step_outputs, ["PhenotypeRepresentation"]), chunkable=True),
            _CommSection("differential_diagnosis", self._tool_group_text(step_outputs, ["DifferentialDiagnosis"]), chunkable=True),
            _CommSection("clinical_relevance_ranker", self._tool_group_text(step_outputs, ["ClinicalRelevanceRanker"]), chunkable=True),
            _CommSection("unprocessed_multimodal_raw", json_to_toon(predictor_input.get("multimodal_unprocessed_raw") or {}), chunkable=True),
            _CommSection("multimodal_narrative", self._tool_group_text(step_outputs, ["MultimodalNarrativeCreator"]), chunkable=True),
            _CommSection("unimodal_outputs", self._tool_group_text(step_outputs, ["UnimodalCompressor"]), chunkable=True),
            # Optional non-ranked context
            _CommSection("feature_synthesizer_optional", self._tool_group_text(step_outputs, ["FeatureSynthesizer"]), optional=True),
            _CommSection("processed_raw_low_priority", json_to_toon(predictor_input.get("multimodal_processed_raw_low_priority") or {}), optional=True, low_priority=True),
            _CommSection(
                "prior_context_fill_low_priority",
                json_to_toon((predictor_input.get("context_fill_report") or {}).get("top_added") or []),
                optional=True,
                low_priority=True,
            ),
            _CommSection("fusion_result_snapshot", json_to_toon(fusion_result), optional=True, low_priority=True),
        ]

        context_note = str(report_context_note or "").strip() or "No additional warning context."
        anchors = {
            "target_condition": str(prediction_dict.get("target_condition") or execution_summary.get("target_condition") or "Unknown target"),
            "control_condition": str(control),
            "prediction_output": json_to_toon(prediction_dict),
            "critic_evaluation": json_to_toon(evaluation_dict),
            "execution_summary": json_to_toon(execution_summary or {}),
            "dataflow_summary": json_to_toon((execution_summary or {}).get("dataflow_summary") or (executor_output or {}).get("dataflow_summary") or {}),
            "report_context_note": context_note,
            "user_focus_modalities": str(user_focus_modalities or "").strip(),
            "user_general_instruction": str(user_general_instruction or "").strip(),
        }
        return sections, anchors

    def _fit_sections(
        self,
        sections: Sequence[_CommSection],
        *,
        token_budget: int,
        model_hint: str,
    ) -> Tuple[str, int]:
        selected: List[_CommSection] = []
        used = 0
        for sec in sections:
            if not str(sec.text or "").strip():
                continue
            block = self._render_sections([sec])
            block_tokens = count_tokens(block, model_hint=model_hint)
            if block_tokens + used > token_budget:
                continue
            selected.append(sec)
            used += block_tokens
        return self._render_sections(selected), used

    def _build_chunks(
        self,
        sections: Sequence[_CommSection],
        *,
        chunk_budget: int,
        model_hint: str,
    ) -> List[List[_CommSection]]:
        expanded: List[_CommSection] = []
        for sec in sections:
            text = str(sec.text or "")
            if not text.strip():
                continue
            block_tokens = count_tokens(text, model_hint=model_hint)
            if block_tokens <= max(128, chunk_budget - 1200):
                expanded.append(sec)
                continue
            slice_budget = max(512, chunk_budget - 1500)
            if self._encoder is not None:
                token_ids = self._encoder.encode(text)
                part_idx = 1
                for start in range(0, len(token_ids), slice_budget):
                    end = min(start + slice_budget, len(token_ids))
                    part = self._encoder.decode(token_ids[start:end])
                    expanded.append(
                        _CommSection(
                            name=f"{sec.name}#part{part_idx}",
                            text=part,
                            chunkable=sec.chunkable,
                            optional=sec.optional,
                            low_priority=sec.low_priority,
                        )
                    )
                    part_idx += 1
            else:
                approx_chars_per_token = 4
                approx_total_tokens = max(1, int(len(text) / approx_chars_per_token))
                parts_count = max(1, (approx_total_tokens + slice_budget - 1) // slice_budget)
                char_chunk = max(1, int(len(text) / parts_count))
                for part_idx in range(1, parts_count + 1):
                    char_start = (part_idx - 1) * char_chunk
                    char_end = len(text) if part_idx == parts_count else min(len(text), char_start + char_chunk)
                    part = text[char_start:char_end]
                    expanded.append(
                        _CommSection(
                            name=f"{sec.name}#part{part_idx}",
                            text=part,
                            chunkable=sec.chunkable,
                            optional=sec.optional,
                            low_priority=sec.low_priority,
                        )
                    )

        chunks: List[List[_CommSection]] = []
        current: List[_CommSection] = []
        current_tokens = 0
        for sec in expanded:
            section_block = self._render_sections([sec])
            section_tokens = count_tokens(section_block, model_hint=model_hint)
            if current and (current_tokens + section_tokens > chunk_budget):
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(sec)
            current_tokens += section_tokens
        if current:
            chunks.append(current)
        return chunks

    def _extract_chunk_evidence(
        self,
        *,
        chunks: Sequence[Sequence[_CommSection]],
        anchors: Dict[str, str],
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, Any]]:
        tool = get_tool("ChunkEvidenceExtractor")
        if tool is None:
            raise RuntimeError("ChunkEvidenceExtractor tool not found for communicator chunk fallback.")

        chunk_rows: List[Dict[str, Any]] = []
        total = len(chunks)
        model_hint = self.settings.models.tool_model or "gpt-5"

        def run_one(idx: int, chunk_sections: Sequence[_CommSection]) -> Tuple[int, Dict[str, Any]]:
            chunk_text = self._render_sections(chunk_sections)
            token_est = count_tokens(chunk_text, model_hint=model_hint)
            names = [s.name for s in chunk_sections]
            payload = {
                "chunk_text": chunk_text,
                "target_condition": anchors.get("target_condition", ""),
                "control_condition": anchors.get("control_condition", ""),
                "chunk_index": idx,
                "chunk_total": total,
                "hinted_feature_keys": names,
            }
            out = tool.execute(payload)
            if out.success:
                out_payload = out.output if isinstance(out.output, dict) else {}
                row = {
                    "chunk_index": idx,
                    "chunk_total": total,
                    "token_est": token_est,
                    "source_sections": names,
                    "summary": out_payload.get("summary", ""),
                    "for_case": out_payload.get("for_case", []),
                    "for_control": out_payload.get("for_control", []),
                    "uncertainty_factors": out_payload.get("uncertainty_factors", []),
                    "key_findings": out_payload.get("key_findings", []),
                    "cited_feature_keys": out_payload.get("cited_feature_keys", []),
                }
                return idx, row
            row = {
                "chunk_index": idx,
                "chunk_total": total,
                "token_est": token_est,
                "source_sections": names,
                "summary": "Chunk evidence extraction failed; section payload preserved in source sections.",
                "for_case": [],
                "for_control": [],
                "uncertainty_factors": [out.error or "tool_error"],
                "key_findings": [],
                "cited_feature_keys": names,
            }
            return idx, row

        backend_value = getattr(self.settings.models.backend, "value", self.settings.models.backend)
        is_local = str(backend_value).lower() == "local"
        max_workers = 1 if is_local else min(10, max(1, total))
        if is_local and total > 1:
            print("[Communicator] Local Backend detected: Sequential chunk evidence extraction (max_workers=1)")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(run_one, idx, chunk): (idx, chunk)
                for idx, chunk in enumerate(chunks, 1)
            }
            for fut in as_completed(futures):
                idx, row = fut.result()
                chunk_rows.append(row)
                if status_callback:
                    status_callback(
                        f"Deep phenotype evidence extraction (chunk {idx}/{total}, ~{row.get('token_est', 0)} tokens, {len(row.get('source_sections', []))} sections)"
                    )

        chunk_rows.sort(key=lambda r: int(r.get("chunk_index", 0)))
        return chunk_rows

    def _build_guided_rag_rows(
        self,
        *,
        executor_output: Dict[str, Any],
        anchors: Dict[str, str],
        token_budget: int,
    ) -> List[Dict[str, Any]]:
        predictor_input = executor_output.get("predictor_input", {}) or {}
        multimodal_raw = predictor_input.get("multimodal_unprocessed_raw") or {}
        step_outputs = executor_output.get("step_outputs", {}) or {}
        participant_id = str(executor_output.get("participant_id") or "unknown")
        query = (
            f"Target: {anchors.get('target_condition', '')}\n"
            f"Control: {anchors.get('control_condition', '')}\n"
            f"Focus: {anchors.get('user_focus_modalities', '')}\n"
            f"Guidance: {anchors.get('user_general_instruction', '')}"
        ).strip()

        candidates: List[Dict[str, Any]] = []
        model_hint = self.LLM_MODEL or "gpt-5"

        for feat, text in self._flatten_feature_candidates(multimodal_raw):
            payload_tokens = count_tokens(json.dumps(feat, default=str), model_hint=model_hint)
            if payload_tokens >= token_budget:
                continue
            candidates.append(
                {
                    "payload": feat,
                    "embed_text": text,
                    "tokens": payload_tokens,
                    "key": text,
                    "source_type": "feature_path",
                }
            )

        for sid, out in sorted(step_outputs.items(), key=lambda x: str(x[0])):
            if not isinstance(out, dict):
                continue
            tool_name = out.get("tool_name") or (out.get("_step_meta") or {}).get("tool_name") or "ToolOutput"
            if tool_name not in {"DifferentialDiagnosis", "ClinicalRelevanceRanker", "MultimodalNarrativeCreator", "UnimodalCompressor"}:
                continue
            snippet = truncate_text_by_tokens(
                json_to_toon(out),
                900,
                model_hint=model_hint,
                suffix="",
            )
            if not snippet.strip():
                continue
            payload = {"step_id": sid, "tool_name": tool_name, "snippet": snippet}
            payload_tokens = count_tokens(json.dumps(payload, default=str), model_hint=model_hint)
            if payload_tokens >= token_budget:
                continue
            candidates.append(
                {
                    "payload": payload,
                    "embed_text": f"{tool_name}: {snippet[:1500]}",
                    "tokens": payload_tokens,
                    "key": f"step:{sid}:{tool_name}",
                    "source_type": "participant_context",
                }
            )

        if not candidates:
            return []

        embedding_model = self.settings.models.embedding_model or "text-embedding-3-large"
        try:
            q_emb = self.embedding_store.get_or_create_global(
                text=query,
                model=embedding_model,
                embed_fn=lambda payload, model: self.llm_client.get_embedding(payload, model=model),
                source_type="query",
            )
            use_embeddings = True
        except Exception:
            q_emb = []
            use_embeddings = False

        def normalize_text(s: str) -> str:
            s = str(s or "").lower().replace("_", " ").replace("-", " ")
            s = re.sub(r"[^\w\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        query_norm = normalize_text(query)

        unique_texts = sorted({(c["embed_text"], c["source_type"]) for c in candidates})
        emb_map: Dict[str, List[float]] = {}
        if use_embeddings and unique_texts:
            try:
                max_workers = int(os.getenv("COMPASS_EMBEDDING_MAX_WORKERS", "200"))
            except Exception:
                max_workers = 200
            workers = max(1, min(max_workers, len(unique_texts)))

            def load_embedding(text: str, source_type: str) -> Tuple[str, Optional[List[float]]]:
                try:
                    if source_type == "feature_path":
                        emb = self.embedding_store.get_or_create_global(
                            text=text,
                            model=embedding_model,
                            embed_fn=lambda payload, model: self.llm_client.get_embedding(payload, model=model),
                            source_type="feature_path",
                        )
                    else:
                        emb = self.embedding_store.get_or_create_participant(
                            participant_id=participant_id,
                            text=text,
                            model=embedding_model,
                            embed_fn=lambda payload, model: self.llm_client.get_embedding(payload, model=model),
                            source_type=source_type,
                        )
                    return text, emb
                except Exception:
                    return text, None

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(load_embedding, text, source_type) for text, source_type in unique_texts]
                for fut in as_completed(futures):
                    text, emb = fut.result()
                    if emb is not None:
                        emb_map[text] = emb

        scored: List[Tuple[float, Dict[str, Any]]] = []
        selected_keys = set()
        for cand in candidates:
            text = cand["embed_text"]
            score = 0.0
            if use_embeddings:
                emb = emb_map.get(text)
                if emb is not None:
                    denom = np.linalg.norm(q_emb) * np.linalg.norm(emb)
                    score = float(np.dot(q_emb, emb) / denom) if denom else 0.0
                else:
                    score = difflib.SequenceMatcher(None, query_norm, normalize_text(text)).ratio()
            else:
                score = difflib.SequenceMatcher(None, query_norm, normalize_text(text)).ratio()
            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        rows: List[Dict[str, Any]] = []
        remaining = token_budget
        for score, cand in scored:
            key = cand["key"]
            if key in selected_keys:
                continue
            feat_tokens = int(cand["tokens"])
            if feat_tokens >= remaining:
                continue
            selected_keys.add(key)
            feat = cand["payload"]
            rows.append(
                {
                    "source": "guided_rag",
                    "participant_id": participant_id,
                    "path": key,
                    "score": round(float(score), 4),
                    "feature_payload": feat,
                    "source_type": cand.get("source_type"),
                }
            )
            remaining -= feat_tokens
            if remaining < 300:
                break
        return rows

    def _flatten_feature_candidates(self, data: Any, parents: Optional[List[str]] = None) -> List[Tuple[Dict[str, Any], str]]:
        if parents is None:
            parents = []
        out: List[Tuple[Dict[str, Any], str]] = []
        if isinstance(data, dict):
            leaves = data.get("_leaves")
            if isinstance(leaves, list):
                for leaf in leaves:
                    if not isinstance(leaf, dict):
                        continue
                    name = leaf.get("feature") or leaf.get("field_name") or "unknown"
                    path = [str(p) for p in parents if str(p).strip()]
                    path_text = " <- ".join([str(name), *path[-3:][::-1]])
                    out.append((leaf, path_text))
            for key, value in data.items():
                if key == "_leaves":
                    continue
                if isinstance(value, (dict, list)):
                    out.extend(self._flatten_feature_candidates(value, parents + [str(key)]))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if "feature" in item or "field_name" in item:
                        name = item.get("feature") or item.get("field_name") or "unknown"
                        path = item.get("path_in_hierarchy") or []
                        if not isinstance(path, list):
                            path = []
                        full = [str(name), *[str(p) for p in path[-3:][::-1]]]
                        out.append((item, " <- ".join(full)))
                    else:
                        out.extend(self._flatten_feature_candidates(item, parents))
        return out

    def _render_sections(self, sections: Sequence[_CommSection]) -> str:
        rows = []
        for sec in sections:
            text = str(sec.text or "").strip()
            if not text:
                continue
            rows.append(f"### {sec.name}\n```text\n{text}\n```")
        return "\n\n".join(rows)

    def _render_anchors(self, anchors: Dict[str, str]) -> str:
        blocks = [
            ("Target Label Context", anchors.get("target_condition", "")),
        ]
        control_context = str(anchors.get("control_condition", "") or "").strip()
        if control_context:
            blocks.append(("Comparator Label Context", control_context))
        blocks.extend(
            [
                ("Prediction Output", anchors.get("prediction_output", "")),
                ("Critic Evaluation", anchors.get("critic_evaluation", "")),
                ("Execution Summary", anchors.get("execution_summary", "")),
                ("Dataflow Summary", anchors.get("dataflow_summary", "")),
                ("Final Verdict Context Note", anchors.get("report_context_note", "")),
                ("Clinical Focus Areas (Optional)", anchors.get("user_focus_modalities", "") or "Not provided"),
                ("Additional Guidance (Optional)", anchors.get("user_general_instruction", "") or "Not provided"),
            ]
        )
        rows = [f"### {title}\n```text\n{body}\n```" for title, body in blocks]
        return "\n\n".join(rows)

    def _render_chunk_evidence(self, rows: Sequence[Dict[str, Any]]) -> str:
        if not rows:
            return "### Chunk Evidence\n```text\nNo chunk evidence rows available.\n```"
        return "### Chunk Evidence (Integration pass)\n```text\n" + json_to_toon(list(rows)) + "\n```"

    def _build_prompt_header(self) -> str:
        return "\n".join(
            [
                "You are the COMPASS Communicator Agent.",
                "Write a clinician-grade deep phenotyping report in Markdown.",
                "",
                "CRITICAL RULES:",
                "1) Use only provided evidence. Never hallucinate.",
                "2) If metrics/z-scores are missing, write 'not provided'.",
                "3) Follow optional user guidance when evidence allows, but do not invent data.",
                "4) This is phenotype matching context, not definitive diagnosis.",
                "5) All tables MUST be Markdown tables only.",
                "",
                "REQUIRED OUTPUT STRUCTURE:",
                "- Title and participant/target header",
                "- Executive overview (clinician-friendly)",
                "- Prediction rationale (classification/regression/hierarchical as applicable)",
                "- Technical summary tables",
                "- Domain-wise deep phenotyping",
                "- Data coverage, uncertainty, and limitations",
                "- Appendix with traceable evidence snippets",
            ]
        )

    def _compose_prompt(
        self,
        *,
        base_header: str,
        anchors_text: str,
        evidence_text: str,
        rag_text: str,
    ) -> str:
        blocks = [
            base_header,
            "## Anchors",
            anchors_text,
            "## Evidence Package",
            evidence_text or "No evidence package provided.",
        ]
        if rag_text:
            blocks.extend(["## Guided Retrieval Additions", rag_text])
        blocks.append(
            "Now produce the final Markdown report. Do not output JSON. Do not include code fences in the final report."
        )
        return "\n\n".join(blocks)

    def _build_prompt(
        self,
        prediction: Any,
        evaluation: Any,
        executor_output: Dict[str, Any],
        data_overview: Dict[str, Any],
        execution_summary: Dict[str, Any],
        report_context_note: str = "",
        control_condition: Optional[str] = None,
        user_focus_modalities: str = "",
        user_general_instruction: str = "",
    ) -> str:
        sections, anchors = self._build_sections(
            prediction=prediction,
            evaluation=evaluation,
            executor_output=executor_output,
            data_overview=data_overview,
            execution_summary=execution_summary,
            report_context_note=report_context_note,
            control_condition=control_condition,
            user_focus_modalities=user_focus_modalities,
            user_general_instruction=user_general_instruction,
        )
        model_hint = self.LLM_MODEL or "gpt-5"
        budget = self._communicator_input_threshold()
        base_header = self._build_prompt_header()
        anchors_text = self._render_anchors(anchors)
        base_tokens = count_tokens(base_header + "\n\n" + anchors_text, model_hint=model_hint)
        evidence_budget = max(3000, budget - base_tokens)
        evidence_text, _ = self._fit_sections(sections, token_budget=evidence_budget, model_hint=model_hint)
        prompt = self._compose_prompt(
            base_header=base_header,
            anchors_text=anchors_text,
            evidence_text=evidence_text,
            rag_text="",
        )
        prompt = self._append_runtime_instruction(
            prompt,
            label="Communicator Runtime Instruction",
        )
        return truncate_text_by_tokens(prompt, budget, model_hint=model_hint)
