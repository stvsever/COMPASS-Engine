"""
COMPASS Fusion Layer

Compresses and integrates outputs from multiple tools into unified representations.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...config.settings import get_settings
from ..llm_client import get_llm_client
from ..json_parser import parse_json_response
import tiktoken
import os
import difflib
import re

from ..path_utils import split_node_path, resolve_requested_subtree, path_is_prefix, normalize_segment
from .embedding_store import get_embedding_store
from .multimodal_coverage import feature_key_set

# Define 128k context limit and 90% threshold



logger = logging.getLogger("compass.fusion")


@dataclass
class FusionResult:
    """Result of fusing multiple tool outputs."""
    fused_narrative: str
    domain_summaries: Dict[str, str]
    key_findings: List[Dict[str, Any]]
    cross_modal_patterns: List[Dict[str, Any]]
    evidence_summary: Dict[str, List[str]]
    tokens_used: int
    source_outputs: List[str]
    # New field to signal raw pass-through
    skipped_fusion: bool = False
    raw_multimodal_data: Optional[Dict[str, Any]] = None
    raw_processed_multimodal_data: Optional[Dict[str, Any]] = None
    raw_step_outputs: Optional[Dict[int, Dict[str, Any]]] = None
    context_fill_report: Optional[Dict[str, Any]] = None



# FUSION_PROMPT removed - now loaded by Integrator Agent from agents/prompts/integrator_prompt.txt


class FusionLayer:
    """
    Fuses multiple tool outputs into unified representations.
    
    This layer:
    1. Collects outputs from all executed tool steps
    2. Compresses redundant information
    3. Integrates narratives across modalities
    4. Prepares consolidated input for the Predictor
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.llm_client = get_llm_client()
        self.embedding_store = get_embedding_store()
        # Initialize encoder for token counting (approximate)
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-5")
        except Exception:
            try:
                self.encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                # Offline/no-cache fallback: keep pipeline operational without tiktoken files.
                self.encoder = None
            
        model_hint = getattr(self.settings.models, "predictor_model", None)
        max_ctx = int(self.settings.effective_context_window(model_hint))
        self.threshold = int(0.9 * max_ctx)
        logger.info(f"FusionLayer: Dynamic Threshold set to {self.threshold} (Context: {max_ctx} max)")
            
        logger.info("FusionLayer initialized")

    def _token_len(self, text: Any) -> int:
        raw = str(text or "")
        if not raw:
            return 0
        if self.encoder is not None:
            try:
                return len(self.encoder.encode(raw))
            except Exception:
                pass
        # Conservative heuristic fallback when encoder is unavailable.
        return max(1, int(len(raw) / 4))

    def _embedding_store_metadata(self) -> Dict[str, Any]:
        db_path = str(getattr(self.embedding_store, "db_path", "unknown"))
        fallback_reason = getattr(self.embedding_store, "fallback_reason", None)
        return {
            "path": db_path,
            "fallback_active": bool(fallback_reason),
            "fallback_reason": fallback_reason,
            "in_memory": db_path == ":memory:",
        }
    
    def smart_fuse(
        self,
        step_outputs: Dict[int, Dict[str, Any]],
        hierarchical_deviation: Dict[str, Any],
        non_numerical_data: str,
        multimodal_data: Dict[str, Any],
        target_condition: str,
        control_condition: str,
        system_prompt: str = ""
    ) -> FusionResult:
        """
        Intelligently decide whether to fuse via LLM or pass raw data based on token usage.
        
        Logic:
        1. Identify processed subtrees from step_outputs
        2. Filter multimodal_data to only include UNPROCESSED subtrees
        3. Estimate total tokens (deviation + notes + unprocessed_multimodal + step_outputs)
        4. If < 90% of context window, SKIP FUSION and pass raw.
        5. Else, perform standard fusion.
        """

        print(f"\n[FusionLayer] Smart Fusion initiated. Threshold: {self.threshold:,} tokens")
        
        # 1. Identify processed domains/subtrees (subtree-aware, purely lexical).
        processed_whole_domains: Set[str] = set()
        processed_prefixes_by_domain: Dict[str, Set[Tuple[str, ...]]] = {}

        for output in (step_outputs or {}).values():
            if not output or not isinstance(output, dict):
                continue
            meta = output.get("_step_meta") or {}
            tool_name = meta.get("tool_name") or output.get("tool_name")
            if tool_name != "UnimodalCompressor":
                continue

            input_domains = meta.get("input_domains") or []
            params = meta.get("parameters") or {}

            node_paths = params.get("node_paths") or []
            if not node_paths and "node_path" in params:
                single = params.get("node_path")
                if single:
                    node_paths = [single] if isinstance(single, str) else list(single)

            # Fallback to output domain label if planner didn't specify input_domains.
            if not input_domains:
                dom_label = output.get("domain") or ""
                if dom_label:
                    input_domains = [str(dom_label).split(":")[0]]

            for dom in input_domains:
                dom = str(dom)
                # Normalize inline paths (e.g., "BRAIN_MRI|structural|morphology") to base domain.
                dom_segs = split_node_path(dom)
                if dom_segs:
                    dom = dom_segs[0]
                dom_features = multimodal_data.get(dom)

                if not node_paths:
                    processed_whole_domains.add(dom)
                    continue

                # Only resolve subtrees for flattened feature lists.
                if not isinstance(dom_features, list):
                    # Can't resolve; treat as processed domain to avoid double-counting.
                    processed_whole_domains.add(dom)
                    continue

                for raw_path in node_paths:
                    segs = split_node_path(raw_path) if not isinstance(raw_path, list) else [str(s) for s in raw_path]
                    segs = [s for s in segs if s]
                    if not segs:
                        continue

                    # Strip domain prefix if present.
                    if normalize_segment(segs[0]) == normalize_segment(dom):
                        segs = segs[1:]

                    # Empty => whole domain.
                    if not segs:
                        processed_whole_domains.add(dom)
                        continue

                    resolved = resolve_requested_subtree(dom_features, dom, segs, cutoff=0.60)
                    if resolved is None:
                        # Fail-safe: do NOT exclude raw data if we can't resolve the requested subtree.
                        continue
                    _d, prefix = resolved
                    processed_prefixes_by_domain.setdefault(dom, set()).add(prefix)

        processed_domains = set(processed_whole_domains) | set(processed_prefixes_by_domain.keys())
        print(f"[FusionLayer] Processed domains (subtree-aware): {processed_domains}")

        # 2. Separate multimodal data into PROCESSED (exclude) and UNPROCESSED (keep raw).
        unprocessed_multimodal: Dict[str, Any] = {}
        excluded_features_by_domain: Dict[str, List[dict]] = {}

        for domain, features in (multimodal_data or {}).items():
            # Domains not represented as flattened lists are passed through as-is.
            if not isinstance(features, list):
                if domain not in processed_whole_domains:
                    unprocessed_multimodal[domain] = features
                continue

            # Whole-domain processed: exclude all leaves for this domain.
            if domain in processed_whole_domains:
                excluded_features_by_domain[domain] = [f for f in features if isinstance(f, dict)]
                continue

            prefixes = list(processed_prefixes_by_domain.get(domain, set()))
            if not prefixes:
                # Unprocessed domain: keep everything.
                unprocessed_multimodal[domain] = self._features_to_nested_tree(features)
                continue

            kept: List[dict] = []
            excluded: List[dict] = []
            for feat in features:
                if not isinstance(feat, dict):
                    continue
                path = feat.get("path_in_hierarchy") or []
                if not isinstance(path, list):
                    kept.append(feat)
                    continue

                if any(path_is_prefix(prefix, path) for prefix in prefixes):
                    excluded.append(feat)
                else:
                    kept.append(feat)

            if kept:
                unprocessed_multimodal[domain] = self._features_to_nested_tree(kept)
            if excluded:
                excluded_features_by_domain[domain] = excluded
        
        # Aggregate key tool outputs for pass-through mode.
        findings = []
        domain_data = {}
        for _, output in (step_outputs or {}).items():
            if not output:
                continue
            if "abnormality_patterns" in output:
                findings.extend(output.get("abnormality_patterns") or [])
            if "key_abnormalities" in output:
                findings.extend(output["key_abnormalities"])
            if "key_findings" in output:
                findings.extend(output["key_findings"])
            if "domain" in output:
                if "domain_synthesis" in output:
                    domain_data[output["domain"]] = output["domain_synthesis"]
                elif "summary" in output:
                    domain_data[output["domain"]] = output["summary"]
                elif "clinical_narrative" in output:
                    domain_data[output["domain"]] = output["clinical_narrative"]

        # 3. Predictor-centric payload estimation (instead of fusion-summary estimate).
        pass_through_result = FusionResult(
            fused_narrative="Raw pass-through mode: detailed synthesis skipped in favor of raw data integrity.",
            domain_summaries=domain_data,
            key_findings=findings,
            cross_modal_patterns=[],
            evidence_summary={"for_case": [], "for_control": []},
            tokens_used=0,
            source_outputs=[str(k) for k in (step_outputs or {}).keys()],
            skipped_fusion=True,
            raw_multimodal_data=unprocessed_multimodal,
            raw_processed_multimodal_data=None,
            raw_step_outputs=step_outputs,
            context_fill_report={},
        )
        baseline_payload = self.compress_for_predictor(
            fusion_result=pass_through_result,
            hierarchical_deviation=hierarchical_deviation,
            non_numerical_data=non_numerical_data,
        )
        baseline_payload_tokens = self._token_len(json.dumps(baseline_payload, default=str))
        single_chunk_limit = int(getattr(self.settings.token_budget, "max_agent_input_tokens", 20000) or 20000)

        processed_raw_tree: Dict[str, Any] = {}
        for dom, feats in (excluded_features_by_domain or {}).items():
            if feats:
                processed_raw_tree[dom] = self._features_to_nested_tree(feats)

        processed_payload_tokens = None
        if processed_raw_tree:
            processed_pass_through = FusionResult(
                fused_narrative=pass_through_result.fused_narrative,
                domain_summaries=domain_data,
                key_findings=findings,
                cross_modal_patterns=[],
                evidence_summary={"for_case": [], "for_control": []},
                tokens_used=0,
                source_outputs=[str(k) for k in (step_outputs or {}).keys()],
                skipped_fusion=True,
                raw_multimodal_data=unprocessed_multimodal,
                raw_processed_multimodal_data=processed_raw_tree,
                raw_step_outputs=step_outputs,
                context_fill_report={},
            )
            processed_payload = self.compress_for_predictor(
                fusion_result=processed_pass_through,
                hierarchical_deviation=hierarchical_deviation,
                non_numerical_data=non_numerical_data,
            )
            processed_payload_tokens = self._token_len(json.dumps(processed_payload, default=str))

        print(
            "[FusionLayer] Predictor payload estimate: "
            f"{baseline_payload_tokens:,} tokens (single-chunk limit: {single_chunk_limit:,}, "
            f"fusion threshold: {self.threshold:,})"
        )

        # Keep all information. Add full processed raw if it fits under threshold.
        filled_multimodal = unprocessed_multimodal
        fill_report: Dict[str, Any] = {}
        include_processed_raw = False
        if processed_payload_tokens is not None and processed_payload_tokens <= self.threshold:
            include_processed_raw = True
            fill_report["processed_raw_full_included"] = True
            fill_report["processed_raw_payload_tokens"] = processed_payload_tokens
        elif baseline_payload_tokens < self.threshold:
            remaining_tokens = self.threshold - baseline_payload_tokens
            phenotype_text = self._build_phenotype_text(step_outputs, target_condition)
            filled_multimodal, fill_report = self._fill_context_with_rag(
                remaining_tokens=remaining_tokens,
                candidate_features_by_domain=excluded_features_by_domain,
                current_unprocessed=unprocessed_multimodal,
                phenotype_text=phenotype_text,
            )
            fill_report["processed_raw_full_included"] = False
        else:
            print(
                "[FusionLayer] Raw pass-through payload exceeds fusion threshold; "
                "preserving full data and relying on chunked predictor path."
            )
            fill_report["processed_raw_full_included"] = False

        fill_report = fill_report or {}
        fill_report.setdefault("embedding_store", self._embedding_store_metadata())
        all_feature_keys = feature_key_set(multimodal_data or {})
        unprocessed_feature_keys = feature_key_set(filled_multimodal or {})
        excluded_feature_keys = set(all_feature_keys) - set(unprocessed_feature_keys)

        final_payload_probe = self.compress_for_predictor(
            fusion_result=FusionResult(
                fused_narrative=pass_through_result.fused_narrative,
                domain_summaries=domain_data,
                key_findings=findings,
                cross_modal_patterns=[],
                evidence_summary={"for_case": [], "for_control": []},
                tokens_used=0,
                source_outputs=[str(k) for k in (step_outputs or {}).keys()],
                skipped_fusion=True,
                raw_multimodal_data=filled_multimodal,
                raw_processed_multimodal_data=processed_raw_tree if include_processed_raw else None,
                raw_step_outputs=step_outputs,
                context_fill_report=fill_report,
            ),
            hierarchical_deviation=hierarchical_deviation,
            non_numerical_data=non_numerical_data,
        )
        final_payload_tokens = self._token_len(json.dumps(final_payload_probe, default=str))
        fill_report["predictor_payload_estimate"] = {
            "baseline_tokens": baseline_payload_tokens,
            "final_tokens": final_payload_tokens,
            "processed_raw_tokens": processed_payload_tokens,
            "single_chunk_limit": single_chunk_limit,
            "threshold": self.threshold,
            "chunked_two_pass_required": final_payload_tokens > self.threshold,
            "strategy": "threshold_driven_no_loss_pass_through",
        }
        fill_report["coverage"] = {
            "processed_whole_domains": sorted(list(processed_whole_domains)),
            "processed_prefixes_by_domain": {
                dom: [list(p) for p in sorted(list(prefixes), key=lambda x: (len(x), str(x)))]
                for dom, prefixes in processed_prefixes_by_domain.items()
            },
            "excluded_leaf_counts": {dom: len(feats) for dom, feats in excluded_features_by_domain.items()},
            "unprocessed_leaf_counts": {
                dom: len(self._collect_feature_keys(data)) for dom, data in filled_multimodal.items()
            },
            "all_feature_count": len(all_feature_keys),
            "unprocessed_feature_count": len(unprocessed_feature_keys),
            "processed_feature_count": len(excluded_feature_keys),
            "missing_feature_count": max(0, len(all_feature_keys - (unprocessed_feature_keys | excluded_feature_keys))),
        }

        if final_payload_tokens > self.threshold:
            print("[FusionLayer] Predictor payload exceeds threshold; chunked two-pass route required.")
        else:
            print("[FusionLayer] Predictor payload fits threshold; direct non-core pass-through is allowed.")

        return FusionResult(
            fused_narrative="Raw pass-through mode: detailed synthesis skipped in favor of raw data integrity.",
            domain_summaries=domain_data,
            key_findings=findings,
            cross_modal_patterns=[], # No cross-modal analysis done without LLM
            evidence_summary={"for_case": [], "for_control": []},
            tokens_used=0,
            source_outputs=[str(k) for k in (step_outputs or {}).keys()],
            skipped_fusion=True,
            raw_multimodal_data=filled_multimodal,
            raw_processed_multimodal_data=processed_raw_tree if include_processed_raw else None,
            raw_step_outputs=step_outputs,
            context_fill_report=fill_report,
        )

    
    def fuse(
        self,
        step_outputs: Dict[int, Dict[str, Any]],
        hierarchical_deviation: Dict[str, Any],
        non_numerical_data: str,
        target_condition: str,
        system_prompt: str = "",
        multimodal_data: Optional[Dict[str, Any]] = None
    ) -> FusionResult:
        """
        Fuse all step outputs into unified representation.
        
        Args:
            step_outputs: Map of step_id to output from plan execution
            hierarchical_deviation: Deviation map (always passed through)
            non_numerical_data: Non-numerical data (always passed through)
            target_condition: Prediction target
        
        Returns:
            FusionResult with integrated outputs
        """
        print(f"\n[FusionLayer] Fusing {len(step_outputs)} tool outputs")
        
        # Organize outputs by type
        narratives = []
        findings = []
        domain_data = {}
        
        for step_id, output in step_outputs.items():
            if not output:
                continue
                
            # Extract narratives
            if "clinical_narrative" in output:
                narratives.append(output["clinical_narrative"])
            if "integrated_narrative" in output:
                narratives.append(output["integrated_narrative"])
            
            # Extract findings - support both old (key_abnormalities) and new (abnormality_patterns) format
            if "abnormality_patterns" in output:
                for pattern in output["abnormality_patterns"]:
                    findings.append({
                        "pattern": pattern.get("pattern_name", "Unknown"),
                        "type": pattern.get("pattern_type", "UNKNOWN"),
                        "severity": pattern.get("severity", "UNKNOWN"),
                        "interpretation": pattern.get("clinical_interpretation", ""),
                        "relevance": pattern.get("relevance_score", 0.5)
                    })
            elif "key_abnormalities" in output:
                findings.extend(output["key_abnormalities"])
            if "key_findings" in output:
                findings.extend(output["key_findings"])
            
            # Extract domain summaries - support new domain_synthesis format
            if "domain" in output:
                if "domain_synthesis" in output:
                    domain_data[output["domain"]] = output["domain_synthesis"]
                elif "summary" in output:
                    domain_data[output["domain"]] = output["summary"]
        
        print(f"[FusionLayer] Collected {len(narratives)} narratives, {len(findings)} findings")
        
        # Build description of tool outputs for LLM
        outputs_description = self._build_outputs_description(step_outputs)
        
        # Create fusion prompt
        if not system_prompt:
             # Fallback if no prompt provided (should be provided by Integrator)
             logger.warning("No system_prompt provided to FusionLayer.fuse - using minimal fallback.")
             system_prompt = "You are the Fusion Layer.Fuse these outputs."
             
        prompt = system_prompt.format(tool_outputs_description=outputs_description)
        
        user_prompt = f"""## TOOL OUTPUTS TO FUSE

{outputs_description}

## HIERARCHICAL DEVIATION MAP (MEAN ABSOLUTE HIERARCHICAL DEVIATIONS)
{self._format_deviation_raw(hierarchical_deviation)}

## NON-NUMERICAL DATA (NON-TABULAR DATA)
{non_numerical_data}

## TARGET CONDITION
{target_condition}

## CONTROL CONDITION
{control_condition}

Please fuse these outputs into a unified representation. PRESERVE all clinical notes and deviation data."""
        
        # Call LLM for intelligent fusion with auto-repair retry
        max_retries = 2
        last_error = None
        current_user_prompt = user_prompt
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"[FusionLayer] ⚠ Auto-repair attempt {attempt}/{max_retries}...")
                error_feedback = f"\n\n### PREVIOUS ERROR\nYour previous response failed validation with error: {last_error}\nPlease fix the JSON format and ensure all required fields are present."
                current_user_prompt = user_prompt + error_feedback

            try:
                response = self.llm_client.call_tool(
                    system_prompt=prompt,
                    user_prompt=current_user_prompt
                )
                
                result_json = parse_json_response(response.content)
                break # Success!
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[FusionLayer] Attempt {attempt} failed: {last_error}")
                if attempt == max_retries:
                    print(f"[FusionLayer] ✗ Fusion failed after {max_retries} retries: {last_error}")
                    raise
        
        fusion_result = FusionResult(
            fused_narrative=result_json.get("fused_narrative", ""),
            domain_summaries=result_json.get("domain_summaries", domain_data),
            key_findings=result_json.get("key_findings", findings[:10]),
            cross_modal_patterns=result_json.get("cross_modal_patterns", []),
            evidence_summary=result_json.get("evidence_summary", {"for_case": [], "for_control": []}),
            tokens_used=response.total_tokens,
            source_outputs=[str(k) for k in (step_outputs or {}).keys()]
        )

        
        # --- POST-FUSION BACKFILL LOGIC ---
        # Did we compress too much? If so, fill with RAG.
        # Estimate remaining space based on the actual Predictor payload size (not the fusion call size).
        baseline = FusionResult(
            fused_narrative=fusion_result.fused_narrative,
            domain_summaries=fusion_result.domain_summaries,
            key_findings=fusion_result.key_findings,
            cross_modal_patterns=fusion_result.cross_modal_patterns,
            evidence_summary=fusion_result.evidence_summary,
            tokens_used=fusion_result.tokens_used,
            source_outputs=fusion_result.source_outputs,
            skipped_fusion=False,
            raw_multimodal_data=None,
            raw_step_outputs=fusion_result.raw_step_outputs,
            context_fill_report=None,
        )
        baseline_payload = self.compress_for_predictor(
            fusion_result=baseline,
            hierarchical_deviation=hierarchical_deviation,
            non_numerical_data=non_numerical_data,
        )
        baseline_tokens = self._token_len(json.dumps(baseline_payload, default=str))
        remaining_buffer = self.threshold - baseline_tokens
        
        if remaining_buffer > 2000 and multimodal_data:
            print(f"[FusionLayer] Post-Compression Space Available: {remaining_buffer} tokens. Initiating Smart Backfill.")
            
            phenotype_text = self._build_phenotype_text(step_outputs, target_condition)

            # For backfill, we consider ALL multimodal domains as candidates.
            candidate_features_by_domain: Dict[str, List[dict]] = {}
            for dom, dom_data in multimodal_data.items():
                if isinstance(dom_data, list):
                    candidate_features_by_domain[dom] = [f for f in dom_data if isinstance(f, dict)]
                elif isinstance(dom_data, dict):
                    # Flatten nested UKB format to leaf dicts.
                    flat = [feat for feat, _ in self._flatten_multimodal_features(dom_data, parents=[dom])]
                    candidate_features_by_domain[dom] = [f for f in flat if isinstance(f, dict)]
                else:
                    continue

            rag_filled_data, fill_report = self._fill_context_with_rag(
                remaining_tokens=remaining_buffer,
                candidate_features_by_domain=candidate_features_by_domain,
                current_unprocessed={},  # Start fresh
                phenotype_text=phenotype_text,
            )
            
            # Attach this backfilled data to the FusionResult so compress_for_predictor can include it
            if rag_filled_data:
                 print(f"[FusionLayer] ✓ Smart Backfill successful. Enhancing context for Predictor.")
                 fusion_result.raw_multimodal_data = rag_filled_data
                 fill_report.setdefault("embedding_store", self._embedding_store_metadata())
                 fusion_result.context_fill_report = fill_report
        if fusion_result.context_fill_report is None:
            fusion_result.context_fill_report = {"embedding_store": self._embedding_store_metadata()}
        else:
            fusion_result.context_fill_report.setdefault("embedding_store", self._embedding_store_metadata())
        
        
        print(f"[FusionLayer] ✓ Fusion complete - {len(fusion_result.key_findings)} key findings")
        print(f"[FusionLayer] Evidence direction: {result_json.get('overall_direction', 'UNKNOWN')}")
        
        return fusion_result
    
    def _build_outputs_description(
        self,
        step_outputs: Dict[int, Dict[str, Any]]
    ) -> str:
        """Build text description of all tool outputs."""
        descriptions = []
        
        for step_id, output in step_outputs.items():
            if not output:
                continue
            
            tool_name = output.get("tool_name", f"Step {step_id}")
            
            # Extract key information
            parts = [f"### {tool_name} (Step {step_id})"]
            
            if "clinical_narrative" in output:
                parts.append(f"**Narrative**: {output['clinical_narrative'][:500]}")
            
            if "key_abnormalities" in output:
                abnormalities = output["key_abnormalities"][:5]
                parts.append(f"**Abnormalities**: {abnormalities}")
            
            if "domain" in output:
                parts.append(f"**Domain**: {output['domain']}")
            
            if "confidence" in output:
                parts.append(f"**Confidence**: {output['confidence']}")
            
            descriptions.append("\n".join(parts))
        
        return "\n\n".join(descriptions)
    
    def _summarize_deviation(self, deviation: Dict[str, Any]) -> str:
        """Create brief summary of hierarchical deviation."""
        if not deviation:
            return "No deviation data available"
        
        summary_parts = []
        
        if "domain_summaries" in deviation:
            for domain, summary in deviation["domain_summaries"].items():
                if isinstance(summary, dict):
                    severity = summary.get("severity", "UNKNOWN")
                    summary_parts.append(f"- {domain}: {severity}")
                else:
                    summary_parts.append(f"- {domain}: {str(summary)[:100]}")
        
        return "\n".join(summary_parts) if summary_parts else "Deviation map available but no summaries"
    
    def compress_for_predictor(
        self,
        fusion_result: FusionResult,
        hierarchical_deviation: Dict[str, Any],
        non_numerical_data: str,
        max_tokens: int = 100000
    ) -> Dict[str, Any]:
        """
        Create final representation for Predictor.
        Handles both fused summaries and raw pass-through.
        """
        print(f"[FusionLayer] Preparing for Predictor...")
        
        if fusion_result.skipped_fusion:
            # RAW PASS-THROUGH MODE
            print(f"[FusionLayer] using RAW PASS-THROUGH format")
            compressed = {
                # Signal context
                "mode": "RAW_PASS_THROUGH",
                
                # Raw Data
                "hierarchical_deviation_raw": hierarchical_deviation,
                "non_numerical_data_raw": non_numerical_data,
                # Canonical multimodal key + backward-compatible aliases
                "multimodal_unprocessed_raw": fusion_result.raw_multimodal_data,
                "multimodal_context_boost": fusion_result.raw_multimodal_data,  # legacy alias
                "unprocessed_multimodal_data_raw": fusion_result.raw_multimodal_data,  # legacy alias
                "multimodal_processed_raw_low_priority": fusion_result.raw_processed_multimodal_data,
                
                # Tool Outputs (preserved)
                "key_findings": fusion_result.key_findings,
                "domain_summaries": fusion_result.domain_summaries,
                "tool_outputs_raw": fusion_result.raw_step_outputs,
                "context_fill_report": fusion_result.context_fill_report,
                
                # Placeholder for schema compatibility
                "fused_narrative": "Raw data provided - see raw fields.",
                "evidence_summary": fusion_result.evidence_summary,
                "cross_modal_patterns": [],
            }
        else:
            # COMPRESSED MODE (Legacy)
            print(f"[FusionLayer] using COMPRESSED FUSION format")
            compressed = {
                "mode": "COMPRESSED",
                
                # Tool-derived summaries
                "fused_narrative": fusion_result.fused_narrative[:5000],
                "domain_summaries": fusion_result.domain_summaries,
                "key_findings": fusion_result.key_findings[:15],
                "cross_modal_patterns": fusion_result.cross_modal_patterns[:8],
                "evidence_summary": fusion_result.evidence_summary,
                "context_fill_report": fusion_result.context_fill_report,
                
                # Still pass critical raw data
                "hierarchical_deviation_raw": hierarchical_deviation,
                "non_numerical_data_raw": non_numerical_data,
                
                # Note: No multimodal raw in this mode as it was too big
                # UNLESS: We performed Post-Fusion Backfill
                "multimodal_unprocessed_raw": fusion_result.raw_multimodal_data,
                "multimodal_context_boost": fusion_result.raw_multimodal_data,  # legacy alias
                "unprocessed_multimodal_data_raw": fusion_result.raw_multimodal_data,  # legacy alias
                "multimodal_processed_raw_low_priority": fusion_result.raw_processed_multimodal_data,
            }
        
        print(f"[FusionLayer] ✓ Final predictor input ready")
        return compressed
    
    def _format_deviation_raw(self, deviation: Dict[str, Any]) -> str:
        """Format raw deviation map for inclusion in prompts."""
        import json
        try:
            # User requested large token limits - increased to 25000 chars
            return json.dumps(deviation, indent=2, default=str)[:200000]
        except:
            return str(deviation)[:200000]

    def _build_phenotype_text(self, step_outputs: Dict[int, Dict[str, Any]], target_condition: str) -> str:
        """
        Build phenotype text used for RAG semantic scoring.

        Prefer PhenotypeRepresentation tool output (if present), otherwise fall back
        to `target_condition`.
        """
        try:
            for out in (step_outputs or {}).values():
                if not isinstance(out, dict):
                    continue
                tool_name = out.get("tool_name") or (out.get("_step_meta") or {}).get("tool_name")
                if tool_name != "PhenotypeRepresentation":
                    continue

                phenotype_summary = out.get("phenotype_summary") or ""
                biomarker = out.get("biomarker_signature") or {}
                abnormal_domains = biomarker.get("abnormal_domains") or []
                pattern_interp = biomarker.get("pattern_interpretation") or ""

                parts = [
                    f"Target: {target_condition}",
                    phenotype_summary,
                ]
                if abnormal_domains:
                    parts.append("Abnormal domains: " + ", ".join(str(d) for d in abnormal_domains))
                if pattern_interp:
                    parts.append(pattern_interp)
                text = "\n".join(p for p in parts if p and str(p).strip()).strip()
                return text or str(target_condition)
        except Exception:
            pass

        return str(target_condition)

    def _features_to_nested_tree(self, features: List[dict]) -> Dict[str, Any]:
        """Convert flattened features (with path_in_hierarchy) into a nested dict tree."""
        root: Dict[str, Any] = {}
        for feat in features or []:
            if not isinstance(feat, dict):
                continue
            path = feat.get("path_in_hierarchy") or []
            if not isinstance(path, list):
                path = []
            cur = root
            for seg in path:
                if seg is None:
                    continue
                seg_s = str(seg)
                if not seg_s:
                    continue
                if seg_s not in cur or not isinstance(cur.get(seg_s), dict):
                    cur[seg_s] = {}
                cur = cur[seg_s]
            cur.setdefault("_leaves", []).append(feat)
        return root

    def _insert_feature_into_tree(self, tree: Dict[str, Any], feat: dict) -> None:
        """Insert a single feature dict into an existing nested tree."""
        if not isinstance(tree, dict) or not isinstance(feat, dict):
            return
        path = feat.get("path_in_hierarchy") or []
        if not isinstance(path, list):
            path = []
        cur = tree
        for seg in path:
            if seg is None:
                continue
            seg_s = str(seg)
            if not seg_s:
                continue
            if seg_s not in cur or not isinstance(cur.get(seg_s), dict):
                cur[seg_s] = {}
            cur = cur[seg_s]
        cur.setdefault("_leaves", []).append(feat)

    def _collect_feature_keys(self, domain_data: Any) -> Set[Tuple[str, Tuple[str, ...]]]:
        """Collect feature keys (id/name + path) already present to avoid duplicates."""
        keys: Set[Tuple[str, Tuple[str, ...]]] = set()

        def add_feat(f: dict):
            if not isinstance(f, dict):
                return
            fid = f.get("feature_id") or f.get("field_name") or f.get("feature") or "unknown"
            path = f.get("path_in_hierarchy") or []
            if not isinstance(path, list):
                path = []
            keys.add((str(fid), tuple(str(p) for p in path)))

        def walk(obj: Any):
            if isinstance(obj, list):
                for it in obj:
                    if isinstance(it, dict):
                        # flattened features
                        if "field_name" in it or "feature" in it:
                            add_feat(it)
                        else:
                            walk(it)
                    else:
                        walk(it)
                return
            if isinstance(obj, dict):
                if "_leaves" in obj and isinstance(obj["_leaves"], list):
                    for lf in obj["_leaves"]:
                        add_feat(lf)
                for k, v in obj.items():
                    if k == "_leaves":
                        continue
                    if isinstance(v, (dict, list)):
                        walk(v)

        walk(domain_data)
        return keys

    def _get_cached_embedding(
        self,
        text: str,
        _cache_dir_legacy=None,
        *,
        participant_id: Optional[str] = None,
        source_type: str = "feature_path",
    ) -> List[float]:
        """Get embedding from SQLite cache or generate and persist it."""
        model = self.settings.models.embedding_model or "text-embedding-3-large"

        def _embed(payload: str, embed_model: str) -> List[float]:
            return self.llm_client.get_embedding(payload, model=embed_model)

        if participant_id:
            return self.embedding_store.get_or_create_participant(
                participant_id=participant_id,
                text=text,
                model=model,
                embed_fn=_embed,
                source_type=source_type,
            )
        return self.embedding_store.get_or_create_global(
            text=text,
            model=model,
            embed_fn=_embed,
            source_type=source_type,
        )

    def _fill_context_with_rag(
        self,
        remaining_tokens: int,
        candidate_features_by_domain: Dict[str, List[dict]],
        current_unprocessed: Dict[str, Any],
        phenotype_text: str,
        max_prefilter: int = 2500,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fill remaining context window with highest-relevance raw features using Semantic RAG.
        """
        filled_multimodal: Dict[str, Any] = dict(current_unprocessed or {})
        report: Dict[str, Any] = {
            "added_count": 0,
            "added_tokens": 0,
            "per_domain": {},
            "top_added": [],
            "semantic_backend": "api_embeddings",
        }

        if remaining_tokens <= 0:
            return filled_multimodal, report

        if not candidate_features_by_domain:
            print("  > RAG: No candidates provided for backfill.")
            return filled_multimodal, report

        # Collect existing keys to avoid duplicates.
        present_keys_by_domain: Dict[str, Set[Tuple[str, Tuple[str, ...]]]] = {}
        for dom, dom_data in filled_multimodal.items():
            present_keys_by_domain[dom] = self._collect_feature_keys(dom_data)

        def feature_breadcrumb(dom: str, feat: dict) -> str:
            feat_name = feat.get("feature") or feat.get("field_name") or "unknown"
            path = feat.get("path_in_hierarchy") or []
            if not isinstance(path, list):
                path = []
            parents = [dom] + [str(p) for p in path if p is not None and str(p).strip()]
            context_parents = parents[-3:][::-1]
            return " <- ".join([str(feat_name), *context_parents])

        # 1) Build candidate list with abnormality prefilter.
        candidates: List[Dict[str, Any]] = []
        for dom, feats in candidate_features_by_domain.items():
            if not isinstance(feats, list):
                continue
            for feat in feats:
                if not isinstance(feat, dict):
                    continue

                # Estimate token cost of injecting this feature.
                feat_str = json.dumps(feat, default=str)
                feat_tokens = self._token_len(feat_str)
                if feat_tokens >= remaining_tokens:
                    continue

                z = feat.get("z_score", None)
                try:
                    z_f = float(z) if z is not None else None
                except Exception:
                    z_f = None

                abnormality_score = 0.0
                if z_f is not None:
                    abnormality_score = min(1.0, abs(z_f) / 3.0)

                name = feat.get("feature") or feat.get("field_name") or "unknown"
                candidates.append(
                    {
                        "domain": str(dom),
                        "data": feat,
                        "feature_name": str(name),
                        "text": feature_breadcrumb(str(dom), feat),
                        "tokens": feat_tokens,
                        "abnormality_score": abnormality_score,
                        "z_score": z_f,
                    }
                )

        if not candidates:
            return filled_multimodal, report

        # Pre-filter to reduce embedding calls.
        candidates.sort(key=lambda c: c.get("abnormality_score", 0.0), reverse=True)
        if len(candidates) > max_prefilter:
            candidates = candidates[:max_prefilter]

        # 2) Semantic ranking with embeddings (fallback to lexical similarity if embeddings unavailable).
        target_emb = None
        try:
            target_emb = self._get_cached_embedding(str(phenotype_text), source_type="query")
        except Exception:
            target_emb = None
            report["semantic_backend"] = "lexical_fallback"
        else:
            report["semantic_backend"] = "api_embeddings"

        def norm_free_text(s: str) -> str:
            s = str(s or "").lower().replace("_", " ").replace("-", " ")
            s = re.sub(r"[^\w\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        scored: List[Tuple[float, float, Dict[str, Any]]] = []
        phenotype_norm = norm_free_text(str(phenotype_text)[:2000])

        # Precompute candidate embeddings in parallel (cache-aware).
        candidate_embeddings: Dict[str, Any] = {}
        embedding_errors = 0
        if target_emb is not None:
            unique_texts = sorted({c.get("text") for c in candidates if c.get("text")})
            if unique_texts:
                try:
                    max_workers = int(os.getenv("COMPASS_EMBEDDING_MAX_WORKERS", "200"))
                except Exception:
                    max_workers = 200
                workers = max(1, min(max_workers, len(unique_texts)))
                report["embedding_workers"] = workers

                def load_one(text: str):
                    try:
                        return text, self._get_cached_embedding(text, source_type="feature_path")
                    except Exception:
                        return text, None

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(load_one, text) for text in unique_texts]
                    for fut in as_completed(futures):
                        text, emb = fut.result()
                        if emb is not None:
                            candidate_embeddings[text] = emb
                        else:
                            embedding_errors += 1

                if embedding_errors:
                    report["embedding_errors"] = embedding_errors
                    report["semantic_backend"] = "api_embeddings_with_fallback"

        for cand in candidates:
            semantic_score_norm = 0.0
            semantic_raw = 0.0
            if target_emb is not None:
                try:
                    cand_emb = candidate_embeddings.get(cand["text"])
                    if cand_emb is None:
                        raise ValueError("missing_embedding")
                    denom = (np.linalg.norm(target_emb) * np.linalg.norm(cand_emb))
                    if denom:
                        semantic_raw = float(np.dot(target_emb, cand_emb) / denom)
                    else:
                        semantic_raw = 0.0
                    semantic_score_norm = max(0.0, min(1.0, (semantic_raw + 1.0) / 2.0))
                except Exception:
                    semantic_score_norm = difflib.SequenceMatcher(None, phenotype_norm, norm_free_text(cand["text"])).ratio()
            else:
                semantic_score_norm = difflib.SequenceMatcher(None, phenotype_norm, norm_free_text(cand["text"])).ratio()

            abnormality_score = float(cand.get("abnormality_score", 0.0))
            combined_score = 0.75 * semantic_score_norm + 0.25 * abnormality_score
            scored.append((combined_score, semantic_score_norm, cand))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 3) Greedy fill by combined score.
        added_rows: List[Dict[str, Any]] = []
        remaining = int(remaining_tokens)
        per_domain: Dict[str, int] = {}

        for combined, semantic_norm, cand in scored:
            if cand["tokens"] >= remaining:
                continue

            dom = cand["domain"]
            feat = cand["data"]
            fid = feat.get("feature_id") or feat.get("field_name") or feat.get("feature") or "unknown"
            path = feat.get("path_in_hierarchy") or []
            if not isinstance(path, list):
                path = []
            key = (str(fid), tuple(str(p) for p in path))
            if key in present_keys_by_domain.setdefault(dom, set()):
                continue

            # Ensure domain container is a nested tree.
            existing = filled_multimodal.get(dom)
            if isinstance(existing, list):
                tree = self._features_to_nested_tree([f for f in existing if isinstance(f, dict)])
                filled_multimodal[dom] = tree
                existing = tree
            if existing is None:
                filled_multimodal[dom] = {}
                existing = filled_multimodal[dom]
            if not isinstance(existing, dict):
                # Can't safely insert; skip.
                continue

            self._insert_feature_into_tree(existing, feat)
            present_keys_by_domain[dom].add(key)
            remaining -= int(cand["tokens"])
            per_domain[dom] = per_domain.get(dom, 0) + 1
            report["added_count"] += 1
            report["added_tokens"] += int(cand["tokens"])

            added_rows.append(
                {
                    "domain": dom,
                    "feature_name": cand.get("feature_name"),
                    "path_in_hierarchy": path,
                    "z_score": cand.get("z_score"),
                    "combined_score": round(float(combined), 4),
                    "semantic_score_norm": round(float(semantic_norm), 4),
                    "abnormality_score": round(float(cand.get("abnormality_score", 0.0)), 4),
                }
            )

            if remaining < 200:
                break

        report["per_domain"] = per_domain
        report["top_added"] = added_rows[:50]
        report["remaining_tokens"] = remaining

        if report["added_count"]:
            print(f"[FusionLayer] RAG Context Fill: Added {report['added_count']} features ({report['added_tokens']} tokens).")

        return filled_multimodal, report

    def _flatten_multimodal_features(self, subdomain_data: Any, parents: List[str] = None) -> List[Tuple[Dict, str]]:
        """
        Recursively extract all feature dicts.
        Handles both raw nested UKB format and DataLoader's flattened format.
        """
        if parents is None:
            parents = []
            
        features = []
        if isinstance(subdomain_data, dict):
            # Check if this IS a feature dict (DataLoader style)
            feat_name = subdomain_data.get("feature") or subdomain_data.get("field_name")
            if feat_name:
                # Use provided path if available, otherwise use parents
                hierarchy = subdomain_data.get("path_in_hierarchy") or []
                if not isinstance(hierarchy, list):
                    hierarchy = []
                full_parents = list(parents) + [str(p) for p in hierarchy if p is not None and str(p).strip()]
                context_parents = full_parents[-3:][::-1]
                parts = [feat_name, *context_parents]
                cache_key = " <- ".join(parts)
                features.append((subdomain_data, cache_key))
                return features

            # UKB nested logic (search for _leaves)
            if "_leaves" in subdomain_data and isinstance(subdomain_data["_leaves"], list):
                for leaf in subdomain_data["_leaves"]:
                    if isinstance(leaf, dict):
                        l_name = leaf.get("feature") or leaf.get("field_name")
                        if l_name:
                            context_parents = parents[-3:][::-1]
                            parts = [l_name, *context_parents]
                            cache_key = " <- ".join(parts)
                            features.append((leaf, cache_key))
            
            # Recurse
            for key, value in subdomain_data.items():
                if key != "_leaves" and isinstance(value, (dict, list)):
                    new_parents = parents + [key]
                    features.extend(self._flatten_multimodal_features(value, new_parents))
        
        elif isinstance(subdomain_data, list):
             for item in subdomain_data:
                 features.extend(self._flatten_multimodal_features(item, parents))
                    
        return features
