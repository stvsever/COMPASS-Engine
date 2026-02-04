"""
COMPASS Orchestrator Agent (Clinical Orchestrated Multi-modal Predictive Agentic Support System)

Main planning agent that creates execution plans for processing participant data.
"""

import json
import uuid
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_agent import BaseAgent
from ..config.settings import get_settings
from ..data.models.execution_plan import ExecutionPlan, PlanStep, ToolName
from ..utils.core.data_loader import ParticipantData
from ..utils.validation import validate_execution_plan

logger = logging.getLogger("compass.orchestrator")


class Orchestrator(BaseAgent):
    """
    The Orchestrator creates execution plans for processing participant data.
    
    Input:
    - data_overview.json with domain coverage and token estimates
    - Target condition (neuropsychiatric or neurologic)
    - Token budget constraints
    - Available tools description
    
    Output:
    - Stepwise execution plan covering ALL available data
    - Tool calls with parameters and dependencies
    """
    
    AGENT_NAME = "Orchestrator"
    PROMPT_FILE = "orchestrator_prompt.txt"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings = get_settings()
        
        # Configure LLM params for BaseAgent._call_llm
        self.LLM_MODEL = self.settings.models.orchestrator_model
        self.LLM_MAX_TOKENS = self.settings.models.orchestrator_max_tokens
        self.LLM_TEMPERATURE = self.settings.models.orchestrator_temperature
    
    def execute(
        self,
        participant_data: ParticipantData,
        target_condition: str,
        token_budget: Optional[int] = None,
        previous_feedback: Optional[str] = None,
        iteration: int = 1
    ) -> ExecutionPlan:
        """
        Create an execution plan for the participant.
        
        Args:
            participant_data: Loaded participant data
            target_condition: "neuropsychiatric" or "neurologic"
            token_budget: Optional token budget override
            previous_feedback: Feedback from critic if re-orchestrating
            iteration: Which iteration of orchestration loop
        
        Returns:
            ExecutionPlan with all steps
        """
        self._log_start(f"planning for {target_condition} prediction")
        
        # Build the user prompt
        user_prompt = self._build_prompt(
            participant_data=participant_data,
            target_condition=target_condition,
            token_budget=token_budget or self.settings.token_budget.total_budget,
            previous_feedback=previous_feedback
        )
        
        # Get UI instance for granular updates
        from ..utils.compass_ui import get_ui
        ui = get_ui()
        
        print(f"[Orchestrator] Participant: {participant_data.participant_id}")
        
        # Granular Status Updates (User requested visibility)
        if hasattr(ui, 'enabled') and ui.enabled:
             ui.set_status("Analyzing Domain Coverage...", stage=1)
             time.sleep(0.5) # Slight pause for visual feedback
             
             ui.set_status("Synthesizing Feature Importance...", stage=1) 
             time.sleep(0.5) 
             
             ui.set_status("Constructing Execution Plan...", stage=1)

        print(f"[Orchestrator] Target: {target_condition}")
        print(f"[Orchestrator] Available domains: {participant_data.get_available_domains()}")
        print(f"[Orchestrator] Iteration: {iteration}")
        
        if previous_feedback:
            print(f"[Orchestrator] Previous feedback: {previous_feedback[:200]}...")
        
        # Call LLM with auto-repair parsing
        plan_data = self._call_llm(user_prompt)
        
        # Validate plan
        is_valid, errors = validate_execution_plan(plan_data)
        if not is_valid:
            logger.warning(f"Plan validation issues: {errors}")
            print(f"[Orchestrator] Plan validation warnings: {errors}")
        
        # Convert to ExecutionPlan
        plan = self._parse_plan(
            plan_data=plan_data,
            participant_id=participant_data.participant_id,
            target_condition=target_condition,
            iteration=iteration,
            previous_feedback=previous_feedback
        )

        
        self._log_complete(
            f"{plan.total_steps} steps planned, "
            f"est. {plan.total_estimated_tokens} tokens"
        )
        
        # Print plan summary
        self._print_plan_summary(plan)
        
        return plan
    
    def _build_prompt(
        self,
        participant_data: ParticipantData,
        target_condition: str,
        token_budget: int,
        previous_feedback: Optional[str]
    ) -> str:
        """Build the user prompt for the orchestrator."""
        import tiktoken
        import json
        
        # Format domain coverage
        coverage_lines = []
        for domain, cov in participant_data.data_overview.domain_coverage.items():
            if cov.is_available:
                tokens = f", ~{cov.total_tokens} tokens" if cov.total_tokens else ""
                coverage_lines.append(
                    f"- {domain}: {cov.present_leaves}/{cov.total_leaves} "
                    f"({cov.coverage_percentage:.1f}%{tokens})"
                )
        
        coverage_text = "\n".join(coverage_lines)
        
        # Calculate Data Volume Context
        try:
            encoder = tiktoken.encoding_for_model("gpt-4")
        except:
            encoder = tiktoken.get_encoding("cl100k_base")
            
        # Determine threshold based on backend settings
        from ..config.settings import LLMBackend
        if self.settings.models.backend == LLMBackend.LOCAL:
            max_ctx = self.settings.models.local_max_tokens
            SMART_FUSION_THRESHOLD = int(0.9 * max_ctx)
        else:
            SMART_FUSION_THRESHOLD = 115000 # Default for OpenAI (128k) 
        
        
        # Raw components estimation
        dev_tokens = len(encoder.encode(json.dumps(participant_data.hierarchical_deviation.to_dict() if hasattr(participant_data.hierarchical_deviation, 'to_dict') else {}, default=str)))
        notes_tokens = len(encoder.encode(str(participant_data.non_numerical_data.raw_text) if hasattr(participant_data.non_numerical_data, 'raw_text') else ""))
        
        # Multimodal estimation: Sum of all LeafNode tokens from DomainCoverage? 
        # Or estimate from multimodal_data directly?
        # Using Overview total_tokens is safer/faster if reliable
        total_data_tokens = participant_data.data_overview.total_tokens
        
        # If total_tokens from overview is 0 or low, it might be uncalculated. 
        # Fallback: deviation + notes + overhead
        if total_data_tokens < 500:
             total_data_tokens = dev_tokens + notes_tokens + 2000 # buffer
             
        volume_status = "FIT_FOR_RAW" if total_data_tokens < SMART_FUSION_THRESHOLD else "REQUIRES_COMPRESSION"
        
        volume_context = f"""
## DATA VOLUME CONTEXT (Orchestrator Awareness)
Total Estimated Raw Data: {total_data_tokens:,} tokens (Switch Threshold: {SMART_FUSION_THRESHOLD:,})
Status: {volume_status}

GUIDANCE:
- If FIT_FOR_RAW: The Fusion Layer will likely PASS DATA RAW to the Predictor. You can plan for high-fidelity extraction without aggressive early compression.
- If REQUIRES_COMPRESSION: The Fusion Layer will COMPRESS data. You must plan for efficient summarization.

### STRATEGY FOR HIGH-VOLUME DOMAINS (Token Optimization)
If a specific domain (e.g., BRAIN_MRI, GENOMICS) has >15k tokens, DO NOT process the entire domain in one step. Instead, split it into multiple `UnimodalCompressor` steps using the `node_paths` parameter to target specific subtrees. This ensures the output is detailed and avoids max_token limits.

**Examples of (granular) Subtree Splitting :**

1. **BRAIN_MRI (High Volume)**:
   - *Instead of:* One step for 'BRAIN_MRI'.
   - *Do (Granular Splitting):* 
     - Step W: UnimodalCompressor(domain='BRAIN_MRI', parameters={{'node_paths': ['BRAIN_MRI:Morphologics']}})
     - Step X: UnimodalCompressor(domain='BRAIN_MRI', parameters={{'node_paths': ['BRAIN_MRI:Connectomics:Structural:streamline_count']}})
     - Step Y: UnimodalCompressor(domain='BRAIN_MRI', parameters={{'node_paths': ['BRAIN_MRI:Connectomics:Structural:fractional_anisotropy']}})
     - Step Z: UnimodalCompressor(domain='BRAIN_MRI', parameters={{'node_paths': ['BRAIN_MRI:Connectomics:structural-functional coupling']}})

2. **BIOLOGICAL_ASSAY (High Volume)**:
   - *Do:* 
     - Step A: UnimodalCompressor(domain='BIOLOGICAL_ASSAY', parameters={{'node_paths': ['BIOLOGICAL_ASSAY:proteomics']}})
     - Step B: UnimodalCompressor(domain='BIOLOGICAL_ASSAY', parameters={{'node_paths': ['BIOLOGICAL_ASSAY:NMR_metabolomics']}})
     - Step C: UnimodalCompressor(domain='BIOLOGICAL_ASSAY', parameters={{'node_paths': ['BIOLOGICAL_ASSAY:haematology']}})
     - Step D: UnimodalCompressor(domain='BIOLOGICAL_ASSAY', parameters={{'node_paths': ['BIOLOGICAL_ASSAY:serum_biochemistry']}})

#NOTE:
- BUT remember that you always just need to choose the subtrees you want to be compressed that seem to be of too high volume to be passed in a later final step to the (phenotypic) Predictor Agent. ; can also be primary domain ; think for yourself cs it dependes on the data overview at hand.
- Refer to the available leaves in the DATA OVERVIEW to determine valid paths.

"""

        # Available tools description
        tools_desc = self._get_tools_description()
        
        prompt_parts = [
            "## PARTICIPANT DATA OVERVIEW",
            f"Participant ID: {participant_data.participant_id}",
            f"\n### Domain Coverage:\n{coverage_text}",
            f"\nTotal estimated tokens: {participant_data.data_overview.total_tokens}",
            f"Token budget: {token_budget}",
            volume_context,
            f"\n## PREDICTION TARGET",
            f"Target: {target_condition}",
            f"\n## AVAILABLE TOOLS",
            tools_desc,
        ]
        
        if previous_feedback:
            prompt_parts.extend([
                "\n## PREVIOUS ATTEMPT FEEDBACK",
                "The previous prediction was deemed UNSATISFACTORY by the critic.",
                "Please revise your plan based on this feedback:",
                previous_feedback
            ])
        
        prompt_parts.append("\nPlease create an execution plan to process all available data and generate a prediction.")
        
        prompt_parts.append("""
## OUTPUT FORMAT
Return a JSON object with:
{
  "plan_id": "string",
  "priority_domains": ["domain1", "domain2"],
  "fusion_strategy": "Description of how to combine data",
  "reasoning": "Explanation of the plan",
  "steps": [
    {
      "step_id": 1,
      "tool_name": "ToolName",
      "description": "What to do",
      "reasoning": "Why this step is needed",
      "input_domains": ["domain"],
      "parameters": {},
      "estimated_tokens": 1000,
      "depends_on": []
    }
  ],
  "total_estimated_tokens": int
}
""")
        
        return "\n".join(prompt_parts)
    
    def _get_tools_description(self) -> str:
        """Get formatted description of available tools."""
        tools = [
            ("PhenotypeRepresentation", "Generate comprehensive phenotype representation (EARLY, parallel-safe)"),
            ("AnomalyNarrativeBuilder", "Build narratives from deviation maps (EARLY, parallel-safe)"),
            ("FeatureSynthesizer", "Synthesize feature importance from hierarchy (EARLY, parallel-safe)"),
            ("UnimodalCompressor", "Compress single-domain data into clinical summaries (MID)"),
            ("ClinicalRelevanceRanker", "Rank features by clinical relevance (MID, after FeatureSynthesizer)"),
            ("MultimodalNarrativeCreator", "Create integrated narratives across 2+ domains (MID, after compression)"),
            ("HypothesisGenerator", "Generate biomedical hypotheses for abnormalities (LATE)"),
            ("DifferentialDiagnosis", "Generate differential diagnoses with rule-out logic (LATE, final step)"),
            ("CodeExecutor", "Execute Python code for custom analyses (FLEXIBLE)"),
        ]
        
        lines = []
        for name, desc in tools:
            lines.append(f"- **{name}**: {desc}")
        
        return "\n".join(lines)
    
    def _parse_plan(
        self,
        plan_data: Dict[str, Any],
        participant_id: str,
        target_condition: str,
        iteration: int,
        previous_feedback: Optional[str]
    ) -> ExecutionPlan:
        """Parse LLM response into ExecutionPlan object."""
        # Parse steps
        steps = []
        for step_data in plan_data.get("steps", []):
            # Get tool name
            tool_name_str = step_data.get("tool_name", "")
            try:
                tool_name = ToolName(tool_name_str)
            except ValueError:
                logger.warning(f"Unknown tool name: {tool_name_str}")
                continue
            
            step = PlanStep(
                step_id=step_data.get("step_id", len(steps) + 1),
                tool_name=tool_name,
                description=step_data.get("description", ""),
                reasoning=step_data.get("reasoning", ""),
                input_domains=step_data.get("input_domains", []),
                parameters=step_data.get("parameters", {}),
                expected_output=step_data.get("expected_output", ""),
                estimated_tokens=step_data.get("estimated_tokens", 0),
                depends_on=step_data.get("depends_on", [])
            )
            steps.append(step)
        
        return ExecutionPlan(
            plan_id=plan_data.get("plan_id", str(uuid.uuid4())[:8]),
            participant_id=participant_id,
            target_condition=target_condition,
            created_at=datetime.now(),
            total_estimated_tokens=plan_data.get("total_estimated_tokens", 0),
            priority_domains=plan_data.get("priority_domains", []),
            fusion_strategy=plan_data.get("fusion_strategy", ""),
            reasoning=plan_data.get("reasoning", ""),
            steps=steps,
            iteration=iteration,
            previous_feedback=previous_feedback
        )
    
    def _print_plan_summary(self, plan: ExecutionPlan):
        """Print a formatted plan summary."""
        print(f"\n[Orchestrator] EXECUTION PLAN: {plan.plan_id}")
        print(f"[Orchestrator] Priority domains: {', '.join(plan.priority_domains)}")
        print(f"[Orchestrator] Steps:")
        
        for step in plan.steps:
            deps = f" (depends on: {step.depends_on})" if step.depends_on else ""
            print(f"  {step.step_id}. {step.tool_name.value}: {step.description[:50]}...{deps}")
        
        print(f"[Orchestrator] Fusion strategy: {plan.fusion_strategy[:100]}...")
