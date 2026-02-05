"""
COMPASS Plan Executor

Executes orchestrator-generated plans step by step.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from ...config.settings import get_settings
from ...data.models.execution_plan import (
    ExecutionPlan,
    PlanStep,
    StepStatus,
    ToolName,
    PlanExecutionResult,
)
from ...tools import get_tool
from .auto_repair import AutoRepair
# Removed duplicate import
from .token_manager import TokenManager
from ...frontend.compass_ui import get_ui
from ..path_utils import split_node_path, resolve_requested_subtree, path_is_prefix

logger = logging.getLogger("compass.plan_executor")


@dataclass
class StepResult:
    """Result of a single step execution."""
    step_id: int
    success: bool
    output: Optional[Dict[str, Any]]
    error: Optional[str]
    tokens_used: int
    execution_time_ms: int


class PlanExecutor:
    """
    Executes orchestrator plans step by step.
    
    Features:
    - Dependency resolution
    - Parallel execution where possible
    - Auto-repair on failures
    - Token tracking
    - Comprehensive logging
    """
    
    def __init__(
        self,
        token_manager: Optional[TokenManager] = None,
        auto_repair: Optional[AutoRepair] = None
    ):
        self.settings = get_settings()
        self.token_manager = token_manager or TokenManager()
        self.auto_repair = auto_repair or AutoRepair()
        self.ui = get_ui()
        
        logger.info("PlanExecutor initialized")
        print("[PlanExecutor] Initialized with auto-repair enabled")
    
    def execute(
        self,
        plan: ExecutionPlan,
        context: Dict[str, Any]
    ) -> PlanExecutionResult:
        """
        Execute a complete plan.
        
        Args:
            plan: ExecutionPlan from orchestrator
            context: Shared context including participant data
        
        Returns:
            PlanExecutionResult with all step outputs
        """
        start_time = time.time()
        logger.info(f"Starting plan execution: {plan.plan_id}")
        print(f"\n{'='*60}")
        print(f"[PlanExecutor] Executing plan: {plan.plan_id}")
        print(f"[PlanExecutor] Total steps: {plan.total_steps}")
        print(f"[PlanExecutor] Priority domains: {plan.priority_domains}")
        print(f"{'='*60}\n")
        
        if self.ui.enabled:
            self.ui.on_plan_created(plan)
        
        # Track outputs from completed steps for dependencies
        step_outputs: Dict[int, Dict[str, Any]] = {}
        
        while not plan.is_complete:
            # Get next executable steps
            executable = plan.get_next_executable_steps()
            
            if not executable:
                # Check if we're stuck
                pending = plan.pending_steps
                if pending:
                    logger.error(f"Stuck with {len(pending)} pending steps")
                    for step in pending:
                        step.mark_failed("Unresolvable dependencies")
                break
            
            # Execute steps in parallel where possible
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from ...config.settings import LLMBackend
            
            # Use max_workers=12 as requested, BUT force 1 for Local LLM to save VRAM
            active_workers = 12
            if self.settings.models.backend == LLMBackend.LOCAL:
                print("[PlanExecutor] Local Backend detected: Forcing sequential execution (max_workers=1)")
                active_workers = 1
            
            with ThreadPoolExecutor(max_workers=active_workers) as executor:
                future_to_step = {}
                
                for step in executable:
                    print(f"\n[PlanExecutor] Step {step.step_id}: {step.tool_name.value}")
                    print(f"[PlanExecutor] Description: {step.description}")
                    
                    if self.ui.enabled:
                        parallel_ids = [s.step_id for s in executable if s.step_id != step.step_id]
                        self.ui.on_step_start(step.step_id, step.tool_name.value, step.description, parallel_with=parallel_ids if parallel_ids else None)
                    
                    step_status_lock = True # Simple mutex conceptual flag 
                    step.status = StepStatus.IN_PROGRESS
                    
                    # Submit to thread pool
                    future = executor.submit(self._execute_step, step, context, step_outputs)
                    future_to_step[future] = step
                
                # Process results as they complete
                for future in as_completed(future_to_step):
                    step = future_to_step[future]
                    try:
                        result = future.result()
                        
                        if result.success:
                            step.mark_completed(
                                result.output or {},
                                result.tokens_used,
                                result.execution_time_ms
                            )
                            # Thread-safe dict update (GIL handles this for dicts usually, but safe to be aware)
                            step_outputs[step.step_id] = result.output or {}
                            print(f"[PlanExecutor] ✓ Step {step.step_id} completed ({result.tokens_used} tokens)")
                            
                            if self.ui.enabled:
                                preview = ""
                                output = result.output or {}
                                for key in ['clinical_narrative', 'integrated_narrative', 'diagnosis', 'classification', 'summary']:
                                    if key in output:
                                        if isinstance(output[key], str):
                                            preview = output[key]
                                        elif isinstance(output[key], dict) and 'diagnosis' in output[key]:
                                            preview = output[key]['diagnosis']
                                        break
                                self.ui.on_step_complete(step.step_id, result.tokens_used, result.execution_time_ms, preview)
                        else:
                            # Try auto-repair (SEQUENTIAL within the thread context, but here we are in main thread)
                            # To keep simple: handle repair in main thread after failure
                            if self.auto_repair.can_repair(step, result.error):
                                print(f"[PlanExecutor] Attempting auto-repair for step {step.step_id}")
                                if self.ui.enabled:
                                     self.ui.on_auto_repair(step.step_id, "Attempting repair logic...")
                                
                                # Repair must happen here, synchronously blocks loop but safer for logic
                                repaired_result = self._attempt_repair(step, result.error, context, step_outputs)
                                
                                if repaired_result.success:
                                    step.mark_completed(
                                        repaired_result.output or {},
                                        repaired_result.tokens_used,
                                        repaired_result.execution_time_ms
                                    )
                                    step_outputs[step.step_id] = repaired_result.output or {}
                                    print(f"[PlanExecutor] ✓ Step {step.step_id} repaired successfully")
                                else:
                                    step.mark_failed(repaired_result.error or "Unknown error")
                                    print(f"[PlanExecutor] ✗ Step {step.step_id} failed: {repaired_result.error}")
                            else:
                                step.mark_failed(result.error or "Unknown error")
                                print(f"[PlanExecutor] ✗ Step {step.step_id} failed: {result.error}")
                                if self.ui.enabled:
                                    self.ui.on_step_failed(step.step_id, str(result.error))
                                    
                    except Exception as e:
                        logger.exception(f"Exception exploring step execution future: {e}")
                        step.mark_failed(str(e))
        
        total_time = int((time.time() - start_time) * 1000)
        
        result = PlanExecutionResult.from_plan(plan)
        result.total_execution_time_ms = total_time
        
        print(f"\n{'='*60}")
        print(f"[PlanExecutor] Plan execution complete")
        print(f"[PlanExecutor] Steps completed: {result.steps_completed}/{plan.total_steps}")
        print(f"[PlanExecutor] Total tokens: {result.total_tokens_used}")
        print(f"[PlanExecutor] Total time: {total_time}ms")
        print(f"{'='*60}\n")
        
        return result
    
    def _execute_step(
        self,
        step: PlanStep,
        context: Dict[str, Any],
        previous_outputs: Dict[int, Dict[str, Any]]
    ) -> StepResult:
        """Execute a single plan step."""
        start_time = time.time()
        
        try:
            # Get the tool
            tool = get_tool(step.tool_name)
            
            if tool is None:
                return StepResult(
                    step_id=step.step_id,
                    success=False,
                    output=None,
                    error=f"Tool not found: {step.tool_name}",
                    tokens_used=0,
                    execution_time_ms=0
                )
            
            # Build tool input
            tool_input = self._build_tool_input(step, context, previous_outputs)
            
            # Execute tool
            tool_output = tool.execute(tool_input)
            
            execution_time = int((time.time() - start_time) * 1000)
            
                # Record tool token usage
            if self.token_manager:
                self.token_manager.record_usage(
                    component="executor",
                    prompt_tokens=getattr(tool_output, "prompt_tokens", 0),
                    completion_tokens=getattr(tool_output, "completion_tokens", tool_output.tokens_used),
                    step_id=step.step_id,
                    step_tool=step.tool_name.value,
                    step_reasoning=step.reasoning
                )
            
            # Capture detailed logs if enabled
            detailed_info = {}
            if self.settings.detailed_tool_logging:
                detailed_info = {
                    "raw_input": tool_input,
                    "raw_output": tool_output.to_dict()
                }

            if tool_output.success:
                output_data = tool_output.to_dict()
                if detailed_info:
                    output_data["_detailed_log"] = detailed_info

                # Attach execution metadata for downstream fusion (subtree-aware processing).
                output_data["_step_meta"] = {
                    "step_id": step.step_id,
                    "tool_name": step.tool_name.value,
                    "input_domains": list((tool_input.get("input_domains") or step.input_domains or [])),
                    "parameters": dict(step.parameters or {}),
                    "depends_on": list(step.depends_on or []),
                }
                
                return StepResult(
                    step_id=step.step_id,
                    success=True,
                    output=output_data,
                    error=None,
                    tokens_used=tool_output.tokens_used,
                    execution_time_ms=execution_time
                )
            else:
                return StepResult(
                    step_id=step.step_id,
                    success=False,
                    output=None,
                    error=tool_output.error,
                    tokens_used=tool_output.tokens_used,
                    execution_time_ms=execution_time
                )
                
        except Exception as e:
            logger.exception(f"Step execution error: {e}")
            return StepResult(
                step_id=step.step_id,
                success=False,
                output=None,
                error=str(e),
                tokens_used=0,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def _build_tool_input(
        self,
        step: PlanStep,
        context: Dict[str, Any],
        previous_outputs: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build input for a tool based on step parameters and context."""
        # Truncate large inputs to fit within context limits
        # FeatureSynthesizer and UnimodalCompressor need more data density
        tool_canonical = step.tool_name.value
        is_high_density = tool_canonical in [
            "FeatureSynthesizer",
            "UnimodalCompressor",
            "AnomalyNarrativeBuilder",
            "ClinicalRelevanceRanker",
        ]
        
        max_depth = 10 if is_high_density else 6
        max_children = 2000 if is_high_density else 200

        hierarchical_deviation = self._truncate_for_context(
            context.get("hierarchical_deviation"),
            max_depth=max_depth,  
            max_children=max_children
        )
        
        raw_input_domains = list(step.input_domains or [])
        normalized_domains: List[str] = []
        inline_node_paths: List[Any] = []
        for dom in raw_input_domains:
            if not isinstance(dom, str):
                normalized_domains.append(dom)
                continue
            segs = split_node_path(dom)
            if not segs:
                normalized_domains.append(dom)
                continue
            normalized_domains.append(segs[0])
            if len(segs) > 1:
                # Treat inline path (e.g., "BRAIN_MRI|structural|morphology") as a node_path.
                inline_node_paths.append("|".join(segs))

        tool_input = {
            "step_id": step.step_id,
            "tool_name": step.tool_name.value,
            "parameters": step.parameters.copy(),
            "input_domains": normalized_domains if normalized_domains else raw_input_domains,
            
            # Always include these (truncated for context limits)
            "hierarchical_deviation": hierarchical_deviation,
            "non_numerical_data": context.get("non_numerical_data"),
            "target_condition": context.get("target_condition"),
            "participant_id": context.get("participant_id"),
        }
        
        # Add domain-specific data if specified (truncated)
        if raw_input_domains:
            multimodal_data = context.get("multimodal_data", {})
            domain_data = {}
            
            # Check for node_path(s) parameter to extract subtree
            node_paths = step.parameters.get("node_paths", [])
            # Backward compatibility
            if not node_paths and "node_path" in step.parameters:
                single = step.parameters["node_path"]
                node_paths = [single] if isinstance(single, list) else [single]
            if inline_node_paths:
                node_paths = list(node_paths or []) + inline_node_paths
            
            for domain in (normalized_domains if normalized_domains else raw_input_domains):
                if domain in multimodal_data:
                    source_data = multimodal_data[domain]
                    
                    # Extract specific subtree(s) if requested
                    if node_paths:
                        sliced_data = self._extract_subtrees_robust(source_data, domain, node_paths)
                        if sliced_data:
                            source_data = sliced_data

                    # If the orchestrator selected explicit subtrees, preserve them as fully as possible.
                    # Per user requirement: do not accidentally drop leaf-level details for selected subtrees.
                    if node_paths:
                        truncated_domain = self._truncate_for_context(
                            source_data,
                            max_depth=32,
                            max_children=2000,
                        )
                    else:
                        truncated_domain = self._truncate_for_context(
                            source_data,
                            max_depth=8 if tool_canonical == "UnimodalCompressor" else 5,
                            max_children=2000 if tool_canonical == "UnimodalCompressor" else 200,
                        )
                    domain_data[domain] = truncated_domain
            tool_input["domain_data"] = domain_data
        
        # Add outputs from dependent steps (truncated)
        if step.depends_on:
            dependency_outputs = {}
            is_narrative_fusion = tool_canonical in ["MultimodalNarrativeCreator"]
            for dep_id in step.depends_on:
                if dep_id in previous_outputs:
                    truncated_output = self._truncate_for_context(
                        previous_outputs[dep_id],
                        max_depth=16 if is_narrative_fusion else 5,
                        max_children=2000 if is_narrative_fusion else 200,
                    )
                    dependency_outputs[f"step_{dep_id}"] = truncated_output
            tool_input["dependency_outputs"] = dependency_outputs
        
        return tool_input
    
    def _extract_subtrees_robust(self, data: Any, domain_name: str, paths: List[Any]) -> Any:
        """
        Robustly extract and merge subtrees based on paths.
        Handles domain prefixes, case-insensitivity, and merging.
        """
        if not paths:
            return data

        # Fast path: flattened UKB features (list of dicts with path_in_hierarchy).
        if isinstance(data, list) and any(isinstance(x, dict) and "path_in_hierarchy" in x for x in data):
            return self._extract_subtrees_from_flat_features(data, domain_name, paths)
            
        merged_result = {}
        found_any = False
        
        for raw_path in paths:
            # Normalize path to list of segments
            if isinstance(raw_path, str):
                segments = split_node_path(raw_path)
            elif isinstance(raw_path, list):
                # Allow callers to pass lists like ["BRAIN_MRI", "structural", "hippocampus"].
                segments = []
                for s in raw_path:
                    segments.extend(split_node_path(s))
            else:
                continue
                
            # Filter out empty segments
            segments = [str(s) for s in segments if s]
            if not segments:
                continue
                
            # Strip domain prefix if present (case-insensitive)
            if segments[0].lower() == domain_name.lower():
                segments = segments[1:]
                
            if not segments:
                 # Requested the full domain
                 return data

            # Extract this specific path
            subtree = self._extract_single_path_case_insensitive(data, segments)
            
            if subtree is not None:
                found_any = True
                # Merge into result structure
                # We try to reconstruct the hierarchy for the extracted piece
                current_level = merged_result
                for i, seg in enumerate(segments[:-1]):
                    # We use the segment name as key (preserving case from path or data?)
                    # Ideally we want to preserve data keys, but we don't know them easily without tracking
                    # Simple approach: use segment from path request
                    if seg not in current_level:
                        current_level[seg] = {}
                    current_level = current_level[seg]
                    if not isinstance(current_level, dict):
                         # Conflict: trying to make a dict where a leaf existed. 
                         # Overwrite/Force dict (simple heuristic)
                         current_level = {}
                
                # Leaf of the path
                last_seg = segments[-1]
                current_level[last_seg] = subtree
        
        return merged_result if found_any else data

    def _extract_subtrees_from_flat_features(self, flat: List[dict], domain_name: str, paths: List[Any]) -> Any:
        """
        Extract subtree(s) from DataLoader-flattened UKB features using `path_in_hierarchy`.

        Output is a nested dict with `_leaves` lists to preserve hierarchy losslessly.
        """
        resolved_prefixes: List[Tuple[str, ...]] = []

        for raw_path in paths:
            if isinstance(raw_path, list):
                segs = []
                for s in raw_path:
                    segs.extend(split_node_path(s))
            else:
                segs = split_node_path(raw_path)
            segs = [s for s in segs if s]
            if not segs:
                continue

            # Strip domain prefix if present.
            if segs and segs[0].lower() == str(domain_name).lower():
                segs = segs[1:]

            # Empty => full domain.
            if not segs:
                return flat

            resolved = resolve_requested_subtree(flat, domain_name, segs, cutoff=0.60)
            if resolved is None:
                continue

            _dom, prefix = resolved
            resolved_prefixes.append(prefix)

        if not resolved_prefixes:
            return flat

        # Filter and de-duplicate features across selected prefixes.
        selected: List[dict] = []
        seen = set()
        for feat in flat:
            if not isinstance(feat, dict):
                continue
            path = feat.get("path_in_hierarchy") or []
            if not isinstance(path, list):
                continue

            if any(path_is_prefix(prefix, path) for prefix in resolved_prefixes):
                key = (feat.get("feature_id") or feat.get("field_name") or feat.get("feature"), tuple(path))
                if key in seen:
                    continue
                seen.add(key)
                selected.append(feat)

        if not selected:
            # Fail-safe: never return empty slices.
            return flat

        # Rebuild nested structure to preserve hierarchy.
        root: Dict[str, Any] = {}
        for feat in selected:
            path = feat.get("path_in_hierarchy") or []
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

    def _extract_single_path_case_insensitive(self, data: Any, path: List[str]) -> Any:
        """Traverse data using path segments with case-insensitive matching."""
        current = data
        for segment in path:
            if current is None:
                return None
                
            if isinstance(current, dict):
                # Try exact match
                if segment in current:
                    current = current[segment]
                    continue
                
                # Try case-insensitive
                found = False
                for k, v in current.items():
                    if k.lower() == segment.lower():
                        current = v
                        found = True
                        break
                if not found:
                    return None
            
            elif isinstance(current, list):
                # Handle list of dicts with 'name' property (common in this dataset)
                found = False
                for item in current:
                    if isinstance(item, dict):
                        # check likely name keys
                        for name_key in ['name', 'node_name', 'modality', 'feature']:
                             if name_key in item and str(item[name_key]).lower() == segment.lower():
                                 current = item
                                 found = True
                                 break
                    if found: break
                if not found:
                    return None
            else:
                return None
                
        return current

    def _truncate_for_context(
        self,
        data: Any,
        max_depth: int = 3,
        max_children: int = 10,
        current_depth: int = 0
    ) -> Any:
        """
        Truncate nested data structures to fit within context limits.
        
        Args:
            data: Input data (dict, list, or scalar)
            max_depth: Maximum nesting depth before truncation
            max_children: Maximum children per node/array before truncation
            current_depth: Current traversal depth
        
        Returns:
            Truncated data structure
        """
        if data is None:
            return None
            
        if current_depth >= max_depth:
            if isinstance(data, dict):
                return {"_truncated": True, "_keys": list(data.keys())[:5]}
            elif isinstance(data, list):
                return {"_truncated": True, "_length": len(data), "_sample": data[:2] if data else []}
            return data
        
        if isinstance(data, dict):
            result = {}
            items = list(data.items())[:max_children]
            for key, value in items:
                if key == "children" and isinstance(value, list):
                    # Limit children nodes specifically
                    result[key] = [
                        self._truncate_for_context(child, max_depth, max_children, current_depth + 1)
                        for child in value[:max_children]
                    ]
                    if len(value) > max_children:
                        result["_children_truncated"] = len(value) - max_children
                else:
                    result[key] = self._truncate_for_context(value, max_depth, max_children, current_depth + 1)
            if len(data) > max_children:
                result["_keys_truncated"] = len(data) - max_children
            return result
        
        elif isinstance(data, list):
            if len(data) > max_children:
                truncated = [
                    self._truncate_for_context(item, max_depth, max_children, current_depth + 1)
                    for item in data[:max_children]
                ]
                truncated.append({"_more_items": len(data) - max_children})
                return truncated
            return [
                self._truncate_for_context(item, max_depth, max_children, current_depth + 1)
                for item in data
            ]
        
        # Truncate long strings
        if isinstance(data, str) and len(data) > 5000:
            return data[:5000] + "...[truncated]"
        
        return data
    
    def _attempt_repair(
        self,
        step: PlanStep,
        error: str,
        context: Dict[str, Any],
        previous_outputs: Dict[int, Dict[str, Any]]
    ) -> StepResult:
        """Attempt to repair and retry a failed step."""
        step.retry_count += 1
        step.status = StepStatus.RETRYING
        
        # Get repair suggestions
        repair_context = self.auto_repair.get_repair_context(step, error)
        
        # Modify input based on repair suggestions
        modified_input = self._build_tool_input(step, context, previous_outputs)
        modified_input["repair_context"] = repair_context
        
        # Retry
        return self._execute_step(step, context, previous_outputs)
