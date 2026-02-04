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
        target_condition: str
    ) -> Dict[str, Any]:
        """
        Execute the plan and return fused outputs.
        
        Args:
            plan: ExecutionPlan from Orchestrator
            participant_data: Loaded participant data
            target_condition: Prediction target
        
        Returns:
            Dict with fused outputs ready for Predictor
        """
        self._log_start(f"executing plan {plan.plan_id}")
        
        # Build execution context
        context = self._build_context(participant_data, target_condition)
        
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
        
        integration_output = self.integrator.execute(
            step_outputs=execution_result.step_outputs,
            context=context,
            target_condition=target_condition
        )
        
        fusion_result = integration_output["fusion_result"]
        compressed = integration_output["predictor_input"]
        
        # Build output
        output = {
            "execution_result": execution_result,
            "fusion_result": fusion_result,
            "predictor_input": compressed,
            "step_outputs": execution_result.step_outputs,
            
            # Always include these
            "hierarchical_deviation": context.get("hierarchical_deviation"),
            "non_numerical_data": context.get("non_numerical_data"),
            
            # Metadata
            "plan_id": plan.plan_id,
            "participant_id": participant_data.participant_id,
            "target_condition": target_condition,
            "domains_processed": plan.priority_domains,
            "total_tokens_used": execution_result.total_tokens_used,
        }
        
        self._log_complete(
            f"fused {len(execution_result.step_outputs)} outputs, "
            f"{execution_result.total_tokens_used} tokens used"
        )
        
        return output
    
    def _build_context(
        self,
        participant_data: ParticipantData,
        target_condition: str
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
            multimodal_dict = participant_data.multimodal_data.features
        
        context = {
            "participant_id": participant_data.participant_id,
            "target_condition": target_condition,
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
