"""
COMPASS Orchestrator Agent (Clinical Ontology-driven Multi-modal Predictive Agentic Support System)

Main planning agent that creates execution plans for processing participant data.
"""

import json
import uuid
import logging
import time
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_agent import BaseAgent
from ..config.settings import get_settings
from ..data.models.execution_plan import ExecutionPlan, PlanStep, ToolName
from ..utils.core.data_loader import ParticipantData
from ..utils.token_packer import count_tokens
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
    JSON_EXPECTED_KEYS = ["plan_id", "steps", "target_condition"]
    
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
        control_condition: str,
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
            control_condition=control_condition,
            token_budget=token_budget or self.settings.token_budget.total_budget,
            previous_feedback=previous_feedback
        )
        
        # Get UI instance for granular updates
        from ..frontend.compass_ui import get_ui
        ui = get_ui()
        
        print(f"[Orchestrator] Participant: {participant_data.participant_id}")
        

        # Call LLM with auto-repair parsing
        raw_plan_data = self._call_llm(user_prompt)
        plan_data = self._normalize_plan_data(
            raw_plan_data=raw_plan_data,
            target_condition=target_condition,
        )
        
        # Ensure required fields exist for validation/compatibility
        if isinstance(plan_data, dict) and "target_condition" not in plan_data:
            plan_data["target_condition"] = target_condition

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
            control_condition=control_condition,
            iteration=iteration,
            previous_feedback=previous_feedback
        )
        if plan.total_steps == 0:
            logger.warning("Orchestrator produced zero valid steps; switching to deterministic fallback plan.")
            print("[Orchestrator] ⚠ No valid steps parsed; using deterministic fallback plan.")
            plan = self._build_fallback_plan(
                participant_data=participant_data,
                target_condition=target_condition,
                control_condition=control_condition,
                iteration=iteration,
                previous_feedback=previous_feedback,
            )
        
        # UI Update: Send full plan
        if hasattr(ui, 'enabled') and ui.enabled:
             ui.set_status("Constructing Execution Plan...", stage=1)
             ui.on_plan_created(plan)

        
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
        control_condition: str,
        token_budget: int,
        previous_feedback: Optional[str]
    ) -> str:
        """Build the user prompt for the orchestrator."""
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
        
        max_ctx = int(self.settings.effective_context_window(self.settings.models.predictor_model))
        SMART_FUSION_THRESHOLD = int(0.9 * max_ctx)
        
        
        # Raw components estimation
        dev_tokens = count_tokens(
            json.dumps(
                participant_data.hierarchical_deviation.to_dict()
                if hasattr(participant_data.hierarchical_deviation, "to_dict")
                else {},
                default=str,
            ),
            model_hint=self.settings.models.predictor_model or "gpt-5",
        )
        notes_tokens = count_tokens(
            str(participant_data.non_numerical_data.raw_text)
            if hasattr(participant_data.non_numerical_data, "raw_text")
            else "",
            model_hint=self.settings.models.predictor_model or "gpt-5",
        )
        
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
If a specific domain (e.g., BRAIN_MRI, GENOMICS) often has >5-15k tokens, DO NOT process the entire domain in one step. Instead, split it into multiple `UnimodalCompressor` steps using the `node_paths` parameter to target specific subtrees. This ensures the output is detailed and avoids max_token limits.

**Examples of (granular) Subtree Splitting :**

1. **BRAIN_MRI (High Volume)**:
   - *Instead of:* One step for 'BRAIN_MRI'.
   - *Do (Granular Splitting):* (example:)
     - Step W: UnimodalCompressor(domain='BRAIN_MRI', parameters={{'node_paths': ['BRAIN_MRI:Morphologics']}})
     - Step X: UnimodalCompressor(domain='BRAIN_MRI', parameters={{'node_paths': ['BRAIN_MRI:Connectomics:Structural:streamline_count']}})
     - Step Y: UnimodalCompressor(domain='BRAIN_MRI', parameters={{'node_paths': ['BRAIN_MRI:Connectomics:Structural:fractional_anisotropy']}})
     - Step Z: UnimodalCompressor(domain='BRAIN_MRI', parameters={{'node_paths': ['BRAIN_MRI:Connectomics:structural-functional coupling']}})
     - Step AA: UnimodalCompressor(domain='BRAIN_MRI', parameters={{'node_paths': ['BRAIN_MRI:Connectomics:functional']}})

2. **BIOLOGICAL_ASSAY (High Volume)**:
   - *Do:* 
     - Step A: UnimodalCompressor(domain='BIOLOGICAL_ASSAY', parameters={{'node_paths': ['BIOLOGICAL_ASSAY:proteomics']}})
     - Step B: UnimodalCompressor(domain='BIOLOGICAL_ASSAY', parameters={{'node_paths': ['BIOLOGICAL_ASSAY:NMR_metabolomics']}})
     - Step C: UnimodalCompressor(domain='BIOLOGICAL_ASSAY', parameters={{'node_paths': ['BIOLOGICAL_ASSAY:haematology']}})
     - Step D: UnimodalCompressor(domain='BIOLOGICAL_ASSAY', parameters={{'node_paths': ['BIOLOGICAL_ASSAY:serum_biochemistry']}})

#NOTES:
- BUT remember that you always just need to choose the subtrees you want to be compressed that seem to be of too high volume to be passed in a later final step to the (phenotypic) Predictor Agent. ; can also be primary domain ; think for yourself cs it dependes on the data overview at hand.
- Refer to the available leaves in the DATA OVERVIEW to determine valid paths! ; you NEED to make your plan based on the available data ; under stand this hierarchial structure of the info that can be used ; do not hallucinate ; give correct subtree specification
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
            f"Control: {control_condition}",
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
  "user_facing_explanation": "Concise 2-sentence summary of the plan for the UI",
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
        control_condition: str,
        iteration: int,
        previous_feedback: Optional[str]
    ) -> ExecutionPlan:
        """Parse LLM response into ExecutionPlan object."""
        # Parse steps
        steps = []
        for step_data in plan_data.get("steps", []):
            if not isinstance(step_data, dict):
                logger.warning("Skipping non-dict step payload: %s", type(step_data).__name__)
                continue
            # Get tool name
            tool_name = self._coerce_tool_name(step_data)
            if tool_name is None:
                logger.warning("Unknown tool name in step payload: %s", step_data.get("tool_name"))
                continue

            depends_on_raw = step_data.get("depends_on", [])
            if isinstance(depends_on_raw, (int, str)):
                depends_on_raw = [depends_on_raw]
            depends_on: List[int] = []
            if isinstance(depends_on_raw, list):
                for dep in depends_on_raw:
                    try:
                        depends_on.append(int(dep))
                    except (TypeError, ValueError):
                        continue

            try:
                step_id = int(step_data.get("step_id", len(steps) + 1) or (len(steps) + 1))
            except (TypeError, ValueError):
                step_id = len(steps) + 1

            input_domains = step_data.get("input_domains", [])
            if isinstance(input_domains, str):
                input_domains = [input_domains]
            if not isinstance(input_domains, list):
                input_domains = []

            parameters = step_data.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {}

            try:
                estimated_tokens = int(step_data.get("estimated_tokens", 0) or 0)
            except (TypeError, ValueError):
                estimated_tokens = 0

            step = PlanStep(
                step_id=step_id,
                tool_name=tool_name,
                description=step_data.get("description", ""),
                reasoning=step_data.get("reasoning", ""),
                input_domains=input_domains,
                parameters=parameters,
                expected_output=step_data.get("expected_output", ""),
                estimated_tokens=estimated_tokens,
                depends_on=depends_on,
            )
            steps.append(step)
        
        return ExecutionPlan(
            plan_id=plan_data.get("plan_id", str(uuid.uuid4())[:8]),
            participant_id=participant_id,
            target_condition=target_condition,
            control_condition=control_condition,
            created_at=datetime.now(),
            total_estimated_tokens=plan_data.get("total_estimated_tokens", 0),
            priority_domains=plan_data.get("priority_domains", []),
            fusion_strategy=plan_data.get("fusion_strategy", ""),
            user_facing_explanation=plan_data.get("user_facing_explanation") or f"Execution plan targeting {', '.join(plan_data.get('priority_domains', ['relevant domains']))}.",
            reasoning=plan_data.get("reasoning", ""),
            steps=steps,
            iteration=iteration,
            previous_feedback=previous_feedback
        )

    def _build_fallback_plan(
        self,
        participant_data: ParticipantData,
        target_condition: str,
        control_condition: str,
        iteration: int,
        previous_feedback: Optional[str],
    ) -> ExecutionPlan:
        """Deterministic fallback plan when LLM output is unusable."""
        present_domains = []
        try:
            present_domains = [
                str(name)
                for name, cov in (participant_data.data_overview.domain_coverage or {}).items()
                if getattr(cov, "is_available", False)
            ]
        except Exception:
            present_domains = []

        estimate = 4000
        try:
            total = int(getattr(participant_data.data_overview, "total_tokens", 0) or 0)
            if total > 0:
                estimate = max(2000, min(12000, int(total * 0.2)))
        except Exception:
            pass

        step = PlanStep(
            step_id=1,
            tool_name=ToolName.PHENOTYPE_REPRESENTATION,
            description="Build unified phenotype representation from all available domains.",
            reasoning=(
                "Fallback path to maintain pipeline continuity when orchestrator JSON "
                "schema is invalid."
            ),
            input_domains=present_domains,
            parameters={},
            expected_output="Comprehensive phenotype summary for downstream predictor context.",
            estimated_tokens=estimate,
            depends_on=[],
        )

        return ExecutionPlan(
            plan_id=f"fallback_{str(uuid.uuid4())[:8]}",
            participant_id=participant_data.participant_id,
            target_condition=target_condition,
            control_condition=control_condition,
            created_at=datetime.now(),
            total_estimated_tokens=estimate,
            priority_domains=present_domains[:5],
            fusion_strategy=(
                "Fallback strategy: prioritize phenotype representation and preserve "
                "raw context for predictor."
            ),
            user_facing_explanation=(
                "Used a safe fallback plan because the orchestrator output format was invalid."
            ),
            reasoning="Deterministic fallback activated after invalid orchestrator output.",
            steps=[step],
            iteration=iteration,
            previous_feedback=previous_feedback,
        )

    def _normalize_plan_data(self, raw_plan_data: Any, target_condition: str) -> Dict[str, Any]:
        """
        Normalize LLM output into an execution-plan dictionary shape.

        Handles common model variants:
        - top-level list of steps
        - wrapped object under keys like "plan"/"execution_plan"/"data"
        - malformed steps as single dict
        """
        candidate: Any = raw_plan_data

        if isinstance(candidate, dict):
            for key in ("plan", "execution_plan", "result", "output", "data"):
                nested = candidate.get(key)
                if isinstance(nested, dict) and ("steps" in nested or "plan_id" in nested):
                    candidate = nested
                    break
                if isinstance(nested, list):
                    candidate = {"steps": nested}
                    break

        if isinstance(candidate, list):
            # Some models emit a list of JSON-encoded step strings; decode when possible.
            if candidate and all(isinstance(item, str) for item in candidate):
                recovered_steps: List[Dict[str, Any]] = []
                for item in candidate:
                    text = str(item or "").strip()
                    if not text or not text.startswith("{"):
                        continue
                    try:
                        parsed_item = json.loads(text)
                    except Exception:
                        continue
                    if isinstance(parsed_item, dict):
                        recovered_steps.append(parsed_item)
                if recovered_steps:
                    candidate = recovered_steps
            normalized: Dict[str, Any] = {"steps": candidate}
        elif isinstance(candidate, dict):
            normalized = dict(candidate)
            if "steps" not in normalized and self._looks_like_step_dict(normalized):
                normalized = {"steps": [normalized]}
        else:
            normalized = {"steps": []}

        steps_value = normalized.get("steps", [])
        if isinstance(steps_value, dict):
            steps_value = list(steps_value.values())
        if not isinstance(steps_value, list):
            steps_value = []
        normalized["steps"] = steps_value

        if "plan_id" not in normalized or not str(normalized.get("plan_id", "")).strip():
            normalized["plan_id"] = str(uuid.uuid4())[:8]
        if "target_condition" not in normalized or not str(normalized.get("target_condition", "")).strip():
            normalized["target_condition"] = target_condition
        if "total_estimated_tokens" not in normalized:
            total_estimated = 0
            for step in steps_value:
                if isinstance(step, dict):
                    try:
                        total_estimated += int(step.get("estimated_tokens", 0) or 0)
                    except (TypeError, ValueError):
                        continue
            normalized["total_estimated_tokens"] = total_estimated

        return normalized

    @staticmethod
    def _looks_like_step_dict(data: Dict[str, Any]) -> bool:
        keys = set(data.keys())
        step_markers = {"tool_name", "tool", "description", "parameters", "input_domains"}
        return bool(keys & step_markers)

    def _coerce_tool_name(self, step_data: Dict[str, Any]) -> Optional[ToolName]:
        alias_map: Dict[str, ToolName] = {}
        for tool in ToolName:
            token = re.sub(r"[^a-z0-9]+", "", tool.value.lower())
            alias_map[token] = tool
        # Common shorthand variants emitted by smaller or fast models.
        alias_map.update(
            {
                "unimodalcompressor": ToolName.UNIMODAL_COMPRESSOR,
                "multimodalnarrativecreator": ToolName.MULTIMODAL_NARRATIVE,
                "multimodalnarrative": ToolName.MULTIMODAL_NARRATIVE,
                "hypothesisgenerator": ToolName.HYPOTHESIS_GENERATOR,
                "codeexecutor": ToolName.CODE_EXECUTOR,
                "featuresynthesizer": ToolName.FEATURE_SYNTHESIZER,
                "clinicalrelevanceranker": ToolName.CLINICAL_RANKER,
                "clinicalranker": ToolName.CLINICAL_RANKER,
                "anomalynarrativebuilder": ToolName.ANOMALY_NARRATIVE,
                "phenotyperepresentation": ToolName.PHENOTYPE_REPRESENTATION,
                "differentialdiagnosis": ToolName.DIFFERENTIAL_DIAGNOSIS,
            }
        )

        candidates = [
            step_data.get("tool_name"),
            step_data.get("tool"),
            step_data.get("name"),
        ]
        for raw in candidates:
            token = re.sub(r"[^a-z0-9]+", "", str(raw or "").lower())
            if not token:
                continue
            mapped = alias_map.get(token)
            if mapped is not None:
                return mapped
        return None
    
    def _print_plan_summary(self, plan: ExecutionPlan):
        """Print a formatted plan summary."""
        print(f"\n[Orchestrator] EXECUTION PLAN: {plan.plan_id}")
        print(f"[Orchestrator] Priority domains: {', '.join(plan.priority_domains)}")
        print(f"[Orchestrator] Steps:")
        
        for step in plan.steps:
            deps = f" (depends on: {step.depends_on})" if step.depends_on else ""
            print(f"  {step.step_id}. {step.tool_name.value}: {step.description[:50]}...{deps}")
        
        print(f"[Orchestrator] Fusion strategy: {plan.fusion_strategy[:100]}...")
