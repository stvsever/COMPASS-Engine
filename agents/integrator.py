"""
COMPASS Integrator Agent

Manages the fusion and integration of tool outputs into a unified representation.
Wraps the core FusionLayer logic to provide a consistent agent interface.
Ensures that data flow remains optimal for the Predictor agent, respecting token limits.
"""

import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from ..core.fusion_layer import FusionLayer, FusionResult

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
