"""
COMPASS Executor Agent

Manages the execution of orchestrator plans through the tool pipeline.
"""

import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from .integrator import Integrator
from ..utils.core.plan_executor import PlanExecutor
from ..utils.core.data_loader import ParticipantData
from ..data.models.execution_plan import ExecutionPlan, PlanExecutionResult
from ..utils.core.multimodal_coverage import (
    feature_key_set,
    feature_map_by_key,
    features_by_keys,
)

logger = logging.getLogger("compass.executor")


class Executor(BaseAgent):
    """
    The Executor manages plan execution through the tool pipeline.
    
    Responsibilities:
    - Execute plan steps in dependency order
    - Manage tool calls with auto-repair
    - Collect and organize outputs
    - Fuse outputs into unified representation
    """
    
    AGENT_NAME = "Executor"
    PROMPT_FILE = ""  # Executor doesn't use a prompt directly ; it's rather a wrapper for the PlanExecutor
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plan_executor = PlanExecutor(token_manager=self.token_manager)
        self.integrator = Integrator(token_manager=self.token_manager)
    
    def execute(
        self,
        plan: ExecutionPlan,
        participant_data: ParticipantData,
        target_condition: str,
        control_condition: str
    ) -> Dict[str, Any]:
        """
        Execute the plan and return fused outputs.
        
        Args:
            plan: ExecutionPlan from Orchestrator
            participant_data: Loaded participant data
            target_condition: Prediction target
            control_condition: Control comparator
        
        Returns:
            Dict with fused outputs ready for Predictor
        """
        self._log_start(f"executing plan {plan.plan_id}")
        
        # Build execution context
        context = self._build_context(participant_data, target_condition, control_condition)
        context["iteration"] = getattr(plan, "iteration", 1)
        
        print(f"[Executor] Plan ID: {plan.plan_id}")
        print(f"[Executor] Total steps: {plan.total_steps}")
        
        # Execute plan
        execution_result = self.plan_executor.execute(plan, context)
        
        print(f"[Executor] Steps completed: {execution_result.steps_completed}")
        print(f"[Executor] Steps failed: {execution_result.steps_failed}")
        
        if execution_result.errors:
            print(f"[Executor] Errors encountered:")
            for error in execution_result.errors:
                print(f"  - Step {error['step_id']}: {error['error'][:100]}")
        
        # Fuse outputs via Integrator Agent
        print(f"\n[Executor] Handing step outputs to Integrator Agent...")
        
        from ..frontend.compass_ui import get_ui
        ui = get_ui()
        fusion_step_id = 900 + context.get("iteration", 1) # Pseudo ID matching UI logic
        
        if ui:
            ui.set_status("Integrator fusing outputs...", stage=3)
            # Start visual step
            ui.on_step_start(
                step_id=fusion_step_id,
                tool_name="Fusion Layer",
                description="Integrating domain outputs into unified representation...",
                stage=3,
            )

        try:
            integration_output = self.integrator.execute(
                step_outputs=execution_result.step_outputs,
                context=context,
                target_condition=target_condition,
                control_condition=control_condition,
            )
        except Exception as e:
            if ui:
                ui.on_step_failed(
                    step_id=fusion_step_id, 
                    error=str(e)
                )
            raise e
        
        fusion_result = integration_output["fusion_result"]
        compressed = integration_output["predictor_input"]
        coverage_ledger = self._build_coverage_ledger(
            multimodal_data=context.get("multimodal_data") or {},
            step_outputs=execution_result.step_outputs or {},
            predictor_input=compressed,
        )
        compressed["coverage_ledger"] = coverage_ledger

        chunk_result = self.integrator.extract_chunk_evidence(
            step_outputs=execution_result.step_outputs or {},
            predictor_input=compressed,
            coverage_ledger=coverage_ledger,
            data_overview=context.get("data_overview") or {},
            hierarchical_deviation=context.get("hierarchical_deviation") or {},
            non_numerical_data=context.get("non_numerical_data") or "",
            target_condition=target_condition,
            control_condition=control_condition,
            iteration=int(context.get("iteration") or 1),
        )

        if ui:
            ui.on_step_complete(
                step_id=fusion_step_id,
                tokens=0,
                duration_ms=0,
                preview="Integration complete.",
            )

        # Generate prediction status
        if ui:
            ui.set_status("Predictor analyzing fused data...", stage=4)
        
        # Build output
        output = {
            "execution_result": execution_result,
            "fusion_result": fusion_result,
            "predictor_input": compressed,
            "step_outputs": execution_result.step_outputs,
            "chunk_evidence": chunk_result.get("chunk_evidence") or [],
            "predictor_chunk_count": int(chunk_result.get("predictor_chunk_count") or 0),
            
            # Always include these
            "data_overview": context.get("data_overview"),
            "hierarchical_deviation": context.get("hierarchical_deviation"),
            "non_numerical_data": context.get("non_numerical_data"),
            
            # Metadata
            "plan_id": plan.plan_id,
            "participant_id": participant_data.participant_id,
            "target_condition": target_condition,
            "control_condition": control_condition,
            "domains_processed": plan.priority_domains,
            "total_tokens_used": execution_result.total_tokens_used,
            "coverage_ledger": coverage_ledger,
        }
        
        self._log_complete(
            f"fused {len(execution_result.step_outputs)} outputs, "
            f"{execution_result.total_tokens_used} tokens used"
        )
        
        return output

    def _build_coverage_ledger(
        self,
        multimodal_data: Dict[str, Any],
        step_outputs: Dict[int, Dict[str, Any]],
        predictor_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build no-loss multimodal coverage ledger and enforce forced-raw recovery when needed.
        """
        all_keys = feature_key_set(multimodal_data)
        fmap = feature_map_by_key(multimodal_data)

        processed: set[str] = set()
        per_step: Dict[str, int] = {}
        for step_id, out in (step_outputs or {}).items():
            if not isinstance(out, dict):
                continue
            meta = out.get("_step_meta") or {}
            keys = meta.get("consumed_feature_keys") or []
            if not isinstance(keys, list):
                keys = []
            keys_set = {str(k) for k in keys if isinstance(k, str)}
            if keys_set:
                per_step[str(step_id)] = len(keys_set)
                processed.update(keys_set)

        raw_unprocessed = predictor_input.get("multimodal_unprocessed_raw") or {}
        unprocessed_payload_keys = feature_key_set(raw_unprocessed)

        processed_in_all = set(processed) & set(all_keys)
        covered = processed_in_all | unprocessed_payload_keys
        missing = set(all_keys) - set(covered)

        forced_raw_features: list[str] = []
        if missing:
            # Guarantee no-loss by appending missing features to raw multimodal payload.
            forced_raw_tree = features_by_keys(fmap, sorted(list(missing)))
            existing = predictor_input.get("multimodal_unprocessed_raw")
            predictor_input["multimodal_unprocessed_raw"] = self._merge_trees(
                existing if isinstance(existing, dict) else {},
                forced_raw_tree,
            )
            predictor_input["multimodal_context_boost"] = predictor_input["multimodal_unprocessed_raw"]
            predictor_input["unprocessed_multimodal_data_raw"] = predictor_input["multimodal_unprocessed_raw"]
            forced_raw_features = sorted(list(missing))
            unprocessed_payload_keys = feature_key_set(predictor_input["multimodal_unprocessed_raw"])
            covered = processed_in_all | unprocessed_payload_keys
            missing = set(all_keys) - set(covered)

        ledger = {
            "all_features": sorted(list(all_keys)),
            "processed_features": sorted(list(processed_in_all)),
            "unprocessed_features": sorted(list(set(all_keys) - set(processed_in_all))),
            "covered_features": sorted(list(covered)),
            "missing_features": sorted(list(missing)),
            "forced_raw_features": forced_raw_features,
            "per_step_consumed_counts": per_step,
            "summary": {
                "all_count": len(all_keys),
                "processed_count": len(processed_in_all),
                "unprocessed_count": len(set(all_keys) - set(processed_in_all)),
                "covered_count": len(covered),
                "missing_count": len(missing),
                "forced_raw_count": len(forced_raw_features),
            },
        }
        return ledger

    def _merge_trees(self, base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        """Merge nested domain trees with `_leaves` lists."""
        if not isinstance(base, dict):
            base = {}
        if not isinstance(extra, dict):
            return base

        merged = dict(base)
        for key, val in extra.items():
            if key not in merged:
                merged[key] = val
                continue

            if isinstance(val, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_trees(merged[key], val)
                continue

            if key == "_leaves" and isinstance(val, list) and isinstance(merged.get(key), list):
                # Deduplicate by feature id/name + path.
                seen = set()
                out = []
                for feat in list(merged[key]) + list(val):
                    if not isinstance(feat, dict):
                        continue
                    fid = str(feat.get("feature_id") or feat.get("field_name") or feat.get("feature") or "unknown")
                    path = tuple(str(p) for p in (feat.get("path_in_hierarchy") or []))
                    k = (fid, path)
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(feat)
                merged[key] = out
                continue

            merged[key] = val

        return merged
    
    def _build_context(
        self,
        participant_data: ParticipantData,
        target_condition: str,
        control_condition: str
    ) -> Dict[str, Any]:
        """Build execution context with all required data."""
        # Convert hierarchical deviation to dict for serialization
        hierarchical_deviation_dict = {}
        if hasattr(participant_data.hierarchical_deviation, 'root'):
            hierarchical_deviation_dict = self._serialize_deviation(
                participant_data.hierarchical_deviation
            )
        
        # Get non-numerical data as string
        non_numerical_str = ""
        if hasattr(participant_data.non_numerical_data, 'raw_text'):
            non_numerical_str = participant_data.non_numerical_data.raw_text
        
        # Get multimodal data as dict
        multimodal_dict = {}
        if hasattr(participant_data.multimodal_data, 'features'):
            # `MultimodalData.features` may contain Pydantic FeatureValue objects.
            # Downstream logic (subtree slicing, RAG, JSON serialization) expects plain dicts.
            features_by_domain = participant_data.multimodal_data.features or {}
            for domain_name, domain_features in features_by_domain.items():
                # Legacy formats may already store nested dicts; preserve them.
                if not isinstance(domain_features, list):
                    multimodal_dict[domain_name] = domain_features
                    continue

                serial: list = []
                for feat in domain_features:
                    if isinstance(feat, dict):
                        serial.append(feat)
                    elif hasattr(feat, "model_dump"):  # Pydantic v2
                        serial.append(feat.model_dump())
                    elif hasattr(feat, "dict"):  # Pydantic v1
                        serial.append(feat.dict())
                    elif hasattr(feat, "__dict__"):
                        serial.append(dict(feat.__dict__))
                    else:
                        serial.append({"value": str(feat)})

                multimodal_dict[domain_name] = serial
        
        context = {
            "participant_id": participant_data.participant_id,
            "target_condition": target_condition,
            "control_condition": control_condition,
            "data_overview": self._serialize_data_overview(participant_data.data_overview),
            "hierarchical_deviation": hierarchical_deviation_dict,
            "non_numerical_data": non_numerical_str,
            "multimodal_data": multimodal_dict,
        }
        
        return context
    
    def _serialize_deviation(self, deviation) -> Dict[str, Any]:
        """Serialize hierarchical deviation to dict."""
        result = {
            "participant_id": deviation.participant_id,
            "domain_summaries": deviation.domain_summaries,
        }
        
        # Serialize root node recursively
        if deviation.root:
            result["root"] = self._serialize_node(deviation.root)
        
        return result
    
    def _serialize_node(self, node) -> Dict[str, Any]:
        """Serialize a deviation node to dict."""
        # Handle severity - it's a property that returns SeverityLevel enum
        severity_value = None
        try:
            sev = node.severity  # Call property to get enum
            if sev is not None:
                severity_value = sev.value if hasattr(sev, 'value') else str(sev)
        except Exception:
            pass
        
        return {
            "node_id": node.node_id,
            "node_name": node.node_name,
            "level": node.level,
            "z_score": node.z_score,
            "raw_value": node.raw_value,
            "direction": node.direction.value if node.direction else None,
            "is_leaf": node.is_leaf,
            "severity": severity_value,
            "children": [self._serialize_node(c) for c in node.children]
        }
    
    def _serialize_data_overview(self, overview) -> Dict[str, Any]:
        """Serialize data overview to dict."""
        coverage = {}
        for name, cov in overview.domain_coverage.items():
            coverage[name] = {
                "present_leaves": cov.present_leaves,
                "total_leaves": cov.total_leaves,
                "coverage_percentage": cov.coverage_percentage,
                "total_tokens": cov.total_tokens
            }
        
        return {
            "participant_id": overview.participant_id,
            "domain_coverage": coverage,
            "total_tokens": overview.total_tokens,
            "available_domains": overview.available_domains
        }
