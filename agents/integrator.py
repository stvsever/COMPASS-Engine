"""
COMPASS Integrator Agent

Manages the fusion and integration of tool outputs into a unified representation.
Wraps the core FusionLayer logic to provide a consistent agent interface.
Ensures that data flow remains optimal for the Predictor agent, respecting token limits.
"""

import logging
from typing import Dict, Any, Optional, List, Sequence, Set

from .base_agent import BaseAgent
from ..utils.core.fusion_layer import FusionLayer, FusionResult
from ..utils.core.predictor_input_assembler import PredictorInputAssembler, PredictorSection
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
        return max(1200, min(30000, int(max_tool_input * 0.80)))
        
    def execute(
        self, 
        step_outputs: Dict[int, Dict[str, Any]], 
        context: Dict[str, Any], 
        target_condition: str
    ) -> Dict[str, Any]:
        """
        Execute the integration process.
        
        Args:
            step_outputs: Outputs from execution steps
            context: Execution context containing raw data (deviation, notes, etc.)
            target_condition: The target prediction condition
            
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
        }
        def _is_core(name: str) -> bool:
            base = name.split("#", 1)[0]
            return base in core_names
        chunk_sections = [s for s in sections if not _is_core(s.name)]
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
            }

        total = len(chunks)
        rows: List[Dict[str, Any]] = []

        for idx, chunk_sections in enumerate(chunks, 1):
            if ui and ui.enabled:
                ui.set_status(f"Integration evidence extraction (chunk {idx}/{total})", stage=3)
            row = self._extract_chunk_with_fallback(
                tool=tool,
                assembler=assembler,
                chunk_sections=list(chunk_sections),
                target_condition=target_condition,
                chunk_index=idx,
                chunk_total=total,
            )
            rows.append(row)

        return {
            "chunk_evidence": rows,
            "predictor_chunk_count": len(chunks),
        }

    def _extract_chunk_with_fallback(
        self,
        *,
        tool: Any,
        assembler: PredictorInputAssembler,
        chunk_sections: List[PredictorSection],
        target_condition: str,
        chunk_index: int,
        chunk_total: int,
        depth: int = 0,
    ) -> Dict[str, Any]:
        chunk_text = assembler.chunk_to_text(chunk_sections, chunk_index, chunk_total)
        source_sections = [s.name for s in chunk_sections]
        hinted_keys = self._chunk_feature_keys(chunk_sections)

        output = tool.execute({
            "chunk_text": chunk_text,
            "target_condition": target_condition,
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

        if len(chunk_sections) > 1 and depth < 3:
            mid = len(chunk_sections) // 2
            left = self._extract_chunk_with_fallback(
                tool=tool,
                assembler=assembler,
                chunk_sections=chunk_sections[:mid],
                target_condition=target_condition,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                depth=depth + 1,
            )
            right = self._extract_chunk_with_fallback(
                tool=tool,
                assembler=assembler,
                chunk_sections=chunk_sections[mid:],
                target_condition=target_condition,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
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

    def _chunk_feature_keys(self, sections: Sequence[PredictorSection], max_keys: int = 800) -> List[str]:
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
