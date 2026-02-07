"""
COMPASS Communicator Agent

Creates a deep, clinician-grade phenotyping report (Markdown).
"""

import json
import logging
from typing import Any, Dict, Optional

from .base_agent import BaseAgent
from ..config.settings import get_settings
from ..utils.toon import json_to_toon
from ..utils.token_packer import truncate_text_by_tokens

logger = logging.getLogger("compass.communicator")


class Communicator(BaseAgent):
    """
    The Communicator creates a deep phenotyping report for clinicians/researchers.
    """

    AGENT_NAME = "Communicator"
    PROMPT_FILE = "communicator_prompt.txt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings = get_settings()

        # Configure LLM params
        self.LLM_MODEL = self.settings.models.communicator_model
        self.LLM_MAX_TOKENS = self.settings.models.communicator_max_tokens
        self.LLM_TEMPERATURE = self.settings.models.communicator_temperature

    def execute(
        self,
        prediction: Any,
        evaluation: Any,
        executor_output: Dict[str, Any],
        data_overview: Dict[str, Any],
        execution_summary: Dict[str, Any],
        report_context_note: str = "",
        control_condition: Optional[str] = None,
    ) -> str:
        """
        Generate the deep phenotyping report in Markdown.
        """
        self._log_start("deep phenotyping report generation")

        user_prompt = self._build_prompt(
            prediction=prediction,
            evaluation=evaluation,
            executor_output=executor_output,
            data_overview=data_overview,
            execution_summary=execution_summary,
            report_context_note=report_context_note,
            control_condition=control_condition,
        )

        response = self._call_llm(user_prompt, expect_json=False)
        content = response.get("content", "").strip()

        self._log_complete("deep_phenotype.md created")
        return content

    def _build_prompt(
        self,
        prediction: Any,
        evaluation: Any,
        executor_output: Dict[str, Any],
        data_overview: Dict[str, Any],
        execution_summary: Dict[str, Any],
        report_context_note: str = "",
        control_condition: Optional[str] = None,
    ) -> str:
        max_in = int(getattr(self.settings.token_budget, "max_agent_input_tokens", 20000) or 20000)

        # Budget allocations
        pred_budget = int(max_in * 0.15)
        eval_budget = int(max_in * 0.12)
        fusion_budget = int(max_in * 0.18)
        predictor_input_budget = int(max_in * 0.25)
        overview_budget = int(max_in * 0.08)
        fill_budget = int(max_in * 0.07)
        exec_budget = int(max_in * 0.05)

        def to_dict(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "dict"):
                return obj.dict()
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return obj

        prediction_dict = to_dict(prediction)
        evaluation_dict = to_dict(evaluation)
        fusion_result = executor_output.get("fusion_result")
        fusion_dict = to_dict(fusion_result)
        predictor_input = executor_output.get("predictor_input", {})
        context_fill_report = predictor_input.get("context_fill_report") or executor_output.get("context_fill_report") or {}
        control = (
            control_condition
            or prediction_dict.get("control_condition")
            or executor_output.get("control_condition")
            or "Control condition not provided"
        )

        pred_text = truncate_text_by_tokens(json_to_toon(prediction_dict), pred_budget, model_hint="gpt-5")
        eval_text = truncate_text_by_tokens(json_to_toon(evaluation_dict), eval_budget, model_hint="gpt-5")
        fusion_text = truncate_text_by_tokens(json_to_toon(fusion_dict), fusion_budget, model_hint="gpt-5")
        predictor_input_text = truncate_text_by_tokens(json_to_toon(predictor_input), predictor_input_budget, model_hint="gpt-5")
        overview_text = truncate_text_by_tokens(json_to_toon(data_overview), overview_budget, model_hint="gpt-5")
        fill_text = truncate_text_by_tokens(json_to_toon(context_fill_report), fill_budget, model_hint="gpt-5")
        exec_text = truncate_text_by_tokens(json_to_toon(execution_summary), exec_budget, model_hint="gpt-5")

        context_note = str(report_context_note or "").strip()
        if not context_note:
            context_note = "No additional warning context."

        return "\n".join([
            "You are the COMPASS Communicator Agent.",
            "Your job: write a clinician-grade deep phenotyping report in Markdown.",
            "",
            "CRITICAL RULES:",
            "1) Do NOT hallucinate. Only use information present in the inputs below.",
            "2) If a z-score or metric is missing, explicitly write 'not provided' and do not invent values.",
            "3) Be structured, evidence-grounded, and precise.",
            "4) Start with an accessible overview, then go deep into domain-level phenotyping.",
            "",
            "## REQUIRED OUTPUT STRUCTURE",
            "- Title + patient/target header",
            "- Executive Overview (1-2 paragraphs, readable for clinicians)",
            "- Plain-language verdict rationale (why CASE vs CONTROL)",
            "- Verdict + confidence + uncertainty factors",
            "- Evidence Summary (table of key findings; include z-scores only if provided)",
            "- Technical Summary Tables (Markdown tables only)",
            "Verdict Drivers table: evidence for CASE vs CONTROL, strength, source, feature keys",
            "Uncertainty & Limitations table: missing data, conflicts, coverage gaps",
            "- Domain-by-domain Deep Phenotyping",
            f"- Differential context (target vs control: {control})",
            "- Data coverage & limitations",
            "- Appendix: Traceable evidence excerpts (short, relevant snippets)",
            "",
            "## INPUTS (TOON, do not repeat verbatim; synthesize)",
            "",
            "### Prediction Output",
            f"```text\n{pred_text}\n```",
            "",
            "### Critic Evaluation",
            f"```text\n{eval_text}\n```",
            "",
            "### Fusion Result",
            f"```text\n{fusion_text}\n```",
            "",
            "### Predictor Input (Evidence Snapshot)",
            f"```text\n{predictor_input_text}\n```",
            "",
            "### Data Overview (Coverage)",
            f"```text\n{overview_text}\n```",
            "",
            "### Context Fill Report (RAG Backfill)",
            f"```text\n{fill_text}\n```",
            "",
            "### Execution Summary",
            f"```text\n{exec_text}\n```",
            "",
            "### Final Verdict Context Note",
            f"```text\n{context_note}\n```",
            "",
            "If the context note indicates an unsatisfactory verdict, include a prominent warning banner at the top of the report.",
            "All tables MUST be Markdown tables (no HTML).",
            "",
            "Now produce the final Markdown report. No JSON, no code fences in the output.",
        ])
