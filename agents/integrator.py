"""
COMPASS Integrator Agent

Manages the fusion and integration of tool outputs into a unified representation.
Wraps the core FusionLayer logic to provide a consistent agent interface.
Ensures that data flow remains optimal for the Predictor agent, respecting token limits.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Sequence, Set

from .base_agent import BaseAgent
from ..utils.core.fusion_layer import FusionLayer, FusionResult
from ..utils.core.predictor_input_assembler import PredictorInputAssembler, PredictorSection
from ..utils.token_packer import count_tokens
from ..tools import get_tool
from ..frontend.compass_ui import get_ui

logger = logging.getLogger("compass.integrator")

class Integrator(BaseAgent):
    """
    The Integrator Agent manages the data fusion process.
    
    It serves as a modular wrapper around the FusionLayer, making the decision
    logic for data integration explicit and agent-based.
    
    Responsibilities:
    - Assess inputs from Executor
    - Execute 'smart_fuse' logic (decide whether to fuse or pass-through)
    - Prepare final compressed input for Predictor
    """
    
    AGENT_NAME = "Integrator"
    # Use the externally defined prompt file
    PROMPT_FILE = "integrator_prompt.txt"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The core logic resides in FusionLayer, which we orchestrate here
        self.fusion_layer = FusionLayer()
        
        # Configure LLM params
        self.LLM_MODEL = self.settings.models.integrator_model
        self.LLM_MAX_TOKENS = self.settings.models.integrator_max_tokens
        self.LLM_TEMPERATURE = self.settings.models.integrator_temperature

    def _chunk_budget_tokens(self) -> int:
        max_tool_input = int(getattr(self.settings.token_budget, "max_tool_input_tokens", 20000) or 20000)
        backend_value = getattr(self.settings.models.backend, "value", self.settings.models.backend)
        is_local = str(backend_value).lower() == "local"
        if is_local:
            return max(10000, min(18000, int(max_tool_input * 0.9)))
        return max(30000, min(60000, int(max_tool_input * 2.0)))
        
    def execute(
        self, 
        step_outputs: Dict[int, Dict[str, Any]], 
        context: Dict[str, Any], 
        target_condition: str,
        control_condition: str
    ) -> Dict[str, Any]:
        """
        Execute the integration process.
        
        Args:
            step_outputs: Outputs from execution steps
            context: Execution context containing raw data (deviation, notes, etc.)
            target_condition: The target prediction condition
            control_condition: The control comparator
            
        Returns:
            Dict containing:
            - fusion_result: The rich object with fusion details
            - predictor_input: The final compressed dict ready for the Predictor
        """
        self._log_start(f"processing {len(step_outputs)} outputs")
        
        # Unpack context
        hierarchical_deviation = context.get("hierarchical_deviation", {})
        non_numerical_data = context.get("non_numerical_data", "")
        multimodal_data = context.get("multimodal_data", {})
        
        # 1. Decision & Execution (Smart Fusion)
        # This checks the 90% threshold and decides between Raw vs Compressed
        fusion_result = self.fusion_layer.smart_fuse(
            step_outputs=step_outputs,
            hierarchical_deviation=hierarchical_deviation,
            non_numerical_data=non_numerical_data,
            multimodal_data=multimodal_data,
            target_condition=target_condition,
            control_condition=control_condition,
            system_prompt=self.system_prompt # Pass the prompt loaded by BaseAgent
        )
        
        # 2. Final Packaging for Predictor
        # Ensures the format matches exactly what the Predictor expects
        predictor_input = self.fusion_layer.compress_for_predictor(
            fusion_result=fusion_result,
            hierarchical_deviation=hierarchical_deviation,
            non_numerical_data=non_numerical_data
        )
        
        self._log_complete(f"integration strategy: {'RAW PASS-THROUGH' if fusion_result.skipped_fusion else 'LLM COMPRESSION'}")
        
        return {
            "fusion_result": fusion_result,
            "predictor_input": predictor_input
        }

    def extract_chunk_evidence(
        self,
        *,
        step_outputs: Dict[int, Dict[str, Any]],
        predictor_input: Dict[str, Any],
        coverage_ledger: Dict[str, Any],
        data_overview: Dict[str, Any],
        hierarchical_deviation: Dict[str, Any],
        non_numerical_data: str,
        target_condition: str,
        control_condition: str,
        iteration: int = 1,
    ) -> Dict[str, Any]:
        """
        Build chunk evidence for Predictor using the ChunkEvidenceExtractor tool.
        """
        assembler = PredictorInputAssembler(
            max_chunk_tokens=self._chunk_budget_tokens(),
            model_hint=self.settings.models.tool_model,
        )
        executor_stub = {
            "step_outputs": step_outputs,
            "data_overview": data_overview,
            "hierarchical_deviation": hierarchical_deviation,
            "non_numerical_data": non_numerical_data,
        }
        sections = assembler.build_sections(
            executor_output=executor_stub,
            predictor_input=predictor_input,
            coverage_ledger=coverage_ledger,
        )
        core_names = {
            "non_numerical_data_raw",
            "hierarchical_deviation_raw",
            "data_overview",
            "phenotype_representation",
            "feature_synthesizer",
            "differential_diagnosis",
        }
        processed_raw_names = {"processed_multimodal_raw_low_priority"}
        def _is_core(name: str) -> bool:
            base = name.split("#", 1)[0]
            return base in core_names
        def _is_processed_raw(name: str) -> bool:
            base = name.split("#", 1)[0]
            return base in processed_raw_names

        chunk_candidate_sections = [s for s in sections if not _is_core(s.name) and not _is_processed_raw(s.name)]
        processed_raw_sections = [s for s in sections if _is_processed_raw(s.name)]

        total_candidate_tokens = 0
        for sec in chunk_candidate_sections:
            total_candidate_tokens += count_tokens(sec.text, model_hint=self.settings.models.tool_model)

        # Chunking decision is driven by payload fit excluding processed-raw low priority.
        # Processed-raw may be attached only if there is remaining room, but it must never force chunking.
        payload_estimate = (predictor_input.get("context_fill_report") or {}).get("predictor_payload_estimate") or {}
        threshold = payload_estimate.get("threshold") or getattr(self.fusion_layer, "threshold", None)
        if not isinstance(threshold, int) or threshold <= 0:
            predictor_model = self.settings.models.predictor_model or "gpt-5"
            threshold = int(0.9 * self.settings.effective_context_window(predictor_model))

        payload_tokens: Optional[int] = payload_estimate.get("final_tokens") if isinstance(payload_estimate.get("final_tokens"), int) else None
        payload_without_processed_tokens: Optional[int] = None
        try:
            model_hint = self.settings.models.predictor_model or "gpt-5"
            if payload_tokens is None:
                payload_tokens = count_tokens(json.dumps(predictor_input, default=str), model_hint=model_hint)
            probe_input = dict(predictor_input)
            probe_input.pop("multimodal_processed_raw_low_priority", None)
            payload_without_processed_tokens = count_tokens(json.dumps(probe_input, default=str), model_hint=model_hint)
        except Exception:
            payload_without_processed_tokens = None

        include_processed_raw_direct = (
            bool(processed_raw_sections)
            and isinstance(payload_tokens, int)
            and isinstance(threshold, int)
            and payload_tokens <= threshold
        )
        payload_without_processed_fits = (
            isinstance(payload_without_processed_tokens, int)
            and isinstance(threshold, int)
            and payload_without_processed_tokens <= threshold
        )
        chunk_budget = self._chunk_budget_tokens()
        payload_fits = (
            isinstance(payload_without_processed_tokens, int)
            and isinstance(threshold, int)
            and payload_without_processed_tokens <= threshold
        )

        fallback_noncore_fit = not isinstance(threshold, int) and total_candidate_tokens <= chunk_budget
        if payload_fits or fallback_noncore_fit:
            direct_sections = list(chunk_candidate_sections)
            if include_processed_raw_direct:
                direct_sections.extend(processed_raw_sections)
            direct_text = ""
            if direct_sections:
                direct_text = assembler.chunk_to_text(direct_sections, 1, 1)
            direct_tokens = 0
            if direct_text:
                try:
                    model_hint = self.settings.models.predictor_model or "gpt-5"
                    direct_tokens = count_tokens(direct_text, model_hint=model_hint)
                except Exception:
                    direct_tokens = 0
            return {
                "chunk_evidence": [],
                "predictor_chunk_count": 0,
                "chunking_skipped": True,
                "chunking_reason": (
                    "payload_fit_excluding_processed_raw"
                    if payload_without_processed_fits
                    else "non_core_fit_under_budget"
                ),
                "non_core_context_text": direct_text,
                "non_core_context_tokens": direct_tokens,
                "processed_raw_excluded": bool(processed_raw_sections) and not include_processed_raw_direct,
            }

        chunk_sections = chunk_candidate_sections
        chunks = assembler.build_chunks(chunk_sections)

        tool = get_tool("ChunkEvidenceExtractor")
        if tool is None:
            raise RuntimeError("ChunkEvidenceExtractor tool not found in registry.")

        ui = get_ui()
        if not chunks:
            return {
                "chunk_evidence": [
                    {
                        "chunk_index": 1,
                        "chunk_total": 1,
                        "source_sections": [],
                        "summary": "No chunkable sections; core context handled separately.",
                        "for_case": [],
                        "for_control": [],
                        "uncertainty_factors": ["core_only_context"],
                        "key_findings": [],
                        "cited_feature_keys": [],
                        "retry_depth": 0,
                    }
                ],
                "predictor_chunk_count": 1,
                "chunking_skipped": False,
                "chunking_reason": "no_chunkable_sections",
                "non_core_context_text": "",
                "non_core_context_tokens": total_candidate_tokens,
                "processed_raw_excluded": bool(processed_raw_sections),
            }

        total = len(chunks)
        rows_by_idx: Dict[int, Dict[str, Any]] = {}
        chunk_meta: Dict[int, Dict[str, Any]] = {}
        fusion_step_id = 900 + int(iteration or 1)

        for idx, chunk_sections in enumerate(chunks, 1):
            chunk_text = assembler.chunk_to_text(chunk_sections, idx, total)
            token_est = count_tokens(chunk_text, model_hint=self.settings.models.tool_model)
            section_names = [s.name for s in chunk_sections]
            preview = ", ".join(section_names[:3])
            if len(section_names) > 3:
                preview = f"{preview}, +{len(section_names) - 3} more"
            chunk_meta[idx] = {
                "chunk_text": chunk_text,
                "token_est": token_est,
                "section_count": len(section_names),
                "section_preview": preview,
            }

        if ui and ui.enabled:
            reason = "payload exceeds single-chunk limit" if total > 1 else "single chunk"
            ui.set_status(f"Integration evidence extraction ({total} chunks; {reason})", stage=3)
            ui.on_step_start(
                step_id=fusion_step_id,
                tool_name="Fusion Layer",
                description=f"Integrating domain outputs (chunk evidence enabled: {total} chunks).",
                stage=3,
            )

        from concurrent.futures import ThreadPoolExecutor, as_completed
        backend_value = getattr(self.settings.models.backend, "value", self.settings.models.backend)
        is_local = str(backend_value).lower() == "local"
        max_workers = 1 if is_local else min(10, total)
        if is_local and total > 1:
            print("[Integrator] Local Backend detected: Sequential chunk evidence extraction (max_workers=1)")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}
            for idx, chunk_sections in enumerate(chunks, 1):
                meta = chunk_meta.get(idx, {})
                token_est = int(meta.get("token_est") or 0)
                section_count = int(meta.get("section_count") or 0)
                section_preview = str(meta.get("section_preview") or "").strip()
                future = pool.submit(
                    self._extract_chunk_with_fallback,
                    tool=tool,
                    assembler=assembler,
                    chunk_sections=list(chunk_sections),
                    target_condition=target_condition,
                    control_condition=control_condition,
                    chunk_index=idx,
                    chunk_total=total,
                    chunk_text=meta.get("chunk_text"),
                )
                future_map[future] = (idx, token_est, section_count, section_preview)

            completed = 0
            for future in as_completed(future_map):
                idx, token_est, section_count, section_preview = future_map[future]
                row = future.result()
                rows_by_idx[idx] = row
                completed += 1
                if ui and ui.enabled:
                    ui.set_status(
                        f"Integration evidence extraction ({completed}/{total} chunks done)",
                        stage=3,
                    )
                    chunk_detail = f"Chunk evidence {completed}/{total} (~{token_est} tokens, {section_count} sections)"
                    if section_preview:
                        chunk_detail = f"{chunk_detail} [{section_preview}]"
                    ui.on_step_start(
                        step_id=fusion_step_id,
                        tool_name="Fusion Layer",
                        description=chunk_detail,
                        stage=3,
                    )

        rows = [rows_by_idx[i] for i in sorted(rows_by_idx)]

        return {
            "chunk_evidence": rows,
            "predictor_chunk_count": len(chunks),
            "chunking_skipped": False,
            "chunking_reason": "non_core_exceeds_budget",
            "non_core_context_text": "",
            "non_core_context_tokens": total_candidate_tokens,
            "processed_raw_excluded": bool(processed_raw_sections),
        }

    def _extract_chunk_with_fallback(
        self,
        *,
        tool: Any,
        assembler: PredictorInputAssembler,
        chunk_sections: List[PredictorSection],
        target_condition: str,
        control_condition: str,
        chunk_index: int,
        chunk_total: int,
        chunk_text: Optional[str] = None,
        depth: int = 0,
    ) -> Dict[str, Any]:
        if chunk_text is None:
            chunk_text = assembler.chunk_to_text(chunk_sections, chunk_index, chunk_total)
        source_sections = [s.name for s in chunk_sections]
        hinted_keys = self._chunk_feature_keys(chunk_sections)

        output = tool.execute({
            "chunk_text": chunk_text,
            "target_condition": target_condition,
            "control_condition": control_condition,
            "chunk_index": chunk_index,
            "chunk_total": chunk_total,
            "hinted_feature_keys": hinted_keys,
        })
        if output.success:
            return self._normalize_chunk_evidence(
                payload=output.output,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                source_sections=source_sections,
                fallback_feature_keys=hinted_keys,
                retry_depth=depth,
            )

        backend_value = getattr(self.settings.models.backend, "value", self.settings.models.backend)
        is_local = str(backend_value).lower() == "local"
        max_split_depth = 1 if is_local else 3
        if len(chunk_sections) > 1 and depth < max_split_depth:
            mid = len(chunk_sections) // 2
            left = self._extract_chunk_with_fallback(
                tool=tool,
                assembler=assembler,
                chunk_sections=chunk_sections[:mid],
                target_condition=target_condition,
                control_condition=control_condition,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                chunk_text=None,
                depth=depth + 1,
            )
            right = self._extract_chunk_with_fallback(
                tool=tool,
                assembler=assembler,
                chunk_sections=chunk_sections[mid:],
                target_condition=target_condition,
                control_condition=control_condition,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                chunk_text=None,
                depth=depth + 1,
            )
            return self._merge_chunk_evidence(
                rows=[left, right],
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                source_sections=source_sections,
                fallback_feature_keys=hinted_keys,
                merge_reason="split_sections_retry",
            )

        return self._normalize_chunk_evidence(
            payload={
                "summary": f"Chunk extraction failed: {output.error}",
                "for_case": [],
                "for_control": [],
                "uncertainty_factors": [str(output.error or "unknown_error")],
                "key_findings": [],
                "cited_feature_keys": hinted_keys,
            },
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            source_sections=source_sections,
            fallback_feature_keys=hinted_keys,
            retry_depth=depth,
        )

    def _normalize_chunk_evidence(
        self,
        *,
        payload: Dict[str, Any],
        chunk_index: int,
        chunk_total: int,
        source_sections: List[str],
        fallback_feature_keys: List[str],
        retry_depth: int,
    ) -> Dict[str, Any]:
        def _as_str_list(value: Any) -> List[str]:
            if not isinstance(value, list):
                return []
            return [str(v) for v in value if v is not None]

        cited = _as_str_list(payload.get("cited_feature_keys"))
        if not cited:
            cited = list(fallback_feature_keys)
        else:
            cited = sorted(set(cited) | set(fallback_feature_keys))

        return {
            "chunk_index": chunk_index,
            "chunk_total": chunk_total,
            "source_sections": source_sections,
            "summary": str(payload.get("summary") or "No summary provided").strip(),
            "for_case": _as_str_list(payload.get("for_case")),
            "for_control": _as_str_list(payload.get("for_control")),
            "uncertainty_factors": _as_str_list(payload.get("uncertainty_factors")),
            "key_findings": payload.get("key_findings") if isinstance(payload.get("key_findings"), list) else [],
            "cited_feature_keys": cited,
            "retry_depth": retry_depth,
        }

    def _merge_chunk_evidence(
        self,
        *,
        rows: List[Dict[str, Any]],
        chunk_index: int,
        chunk_total: int,
        source_sections: List[str],
        fallback_feature_keys: List[str],
        merge_reason: str,
    ) -> Dict[str, Any]:
        def _as_str_list(value: Any) -> List[str]:
            if not isinstance(value, list):
                return []
            return [str(v) for v in value if v is not None]

        case_rows: List[str] = []
        control_rows: List[str] = []
        uncertainty_rows: List[str] = []
        findings: List[Dict[str, Any]] = []
        cited: Set[str] = set(fallback_feature_keys)

        for row in rows:
            case_rows.extend(_as_str_list(row.get("for_case")))
            control_rows.extend(_as_str_list(row.get("for_control")))
            uncertainty_rows.extend(_as_str_list(row.get("uncertainty_factors")))
            if isinstance(row.get("key_findings"), list):
                findings.extend([f for f in row.get("key_findings") if isinstance(f, dict)])
            cited.update(_as_str_list(row.get("cited_feature_keys")))

        summary_parts = [str(r.get("summary") or "").strip() for r in rows if str(r.get("summary") or "").strip()]
        summary = " | ".join(summary_parts) if summary_parts else "Merged chunk evidence."

        return {
            "chunk_index": chunk_index,
            "chunk_total": chunk_total,
            "source_sections": source_sections,
            "summary": summary,
            "for_case": list(dict.fromkeys(case_rows)),
            "for_control": list(dict.fromkeys(control_rows)),
            "uncertainty_factors": list(dict.fromkeys(uncertainty_rows + [f"merge_reason:{merge_reason}"])),
            "key_findings": findings[:20],
            "cited_feature_keys": sorted(cited),
            "retry_depth": max(int(r.get("retry_depth", 0) or 0) for r in rows) if rows else 0,
        }

    def _chunk_feature_keys(self, sections: Sequence[PredictorSection], max_keys: int = 120) -> List[str]:
        keys: List[str] = []
        seen: Set[str] = set()
        for sec in sections:
            for key in sec.feature_keys:
                if key in seen:
                    continue
                seen.add(key)
                keys.append(str(key))
                if len(keys) >= max_keys:
                    return keys
        return keys
