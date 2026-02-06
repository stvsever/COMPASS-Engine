"""
COMPASS Communicator Agent

Creates a deep, clinician-grade phenotyping report (Markdown).
"""

import json
import logging
from typing import Any, Dict, Optional

from .base_agent import BaseAgent
from ..config.settings import get_settings
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

        pred_text = truncate_text_by_tokens(json.dumps(prediction_dict, indent=2, default=str), pred_budget, model_hint="gpt-5")
        eval_text = truncate_text_by_tokens(json.dumps(evaluation_dict, indent=2, default=str), eval_budget, model_hint="gpt-5")
        fusion_text = truncate_text_by_tokens(json.dumps(fusion_dict, indent=2, default=str), fusion_budget, model_hint="gpt-5")
        predictor_input_text = truncate_text_by_tokens(json.dumps(predictor_input, indent=2, default=str), predictor_input_budget, model_hint="gpt-5")
        overview_text = truncate_text_by_tokens(json.dumps(data_overview, indent=2, default=str), overview_budget, model_hint="gpt-5")
        fill_text = truncate_text_by_tokens(json.dumps(context_fill_report, indent=2, default=str), fill_budget, model_hint="gpt-5")
        exec_text = truncate_text_by_tokens(json.dumps(execution_summary, indent=2, default=str), exec_budget, model_hint="gpt-5")

        return "\n".join([
            "You are the COMPASS Communicator Agent.",
            "Your job: write a clinician-grade DEEP PHENOTYPING report (in Markdown).",
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
            "- Verdict + confidence + uncertainty factors",
            "- Evidence Summary (table of key findings; include z-scores only if provided)",
            "- Domain-by-domain Deep Phenotyping",
            "- Differential context (target vs non-psychiatric control)",
            "- Data coverage & limitations",
            "- Appendix: Traceable evidence excerpts (short, relevant snippets)",
            "",
            "## INPUTS (JSON, do not repeat verbatim; synthesize)",
            "",
            "### Prediction Output",
            f"```json\n{pred_text}\n```",
            "",
            "### Critic Evaluation",
            f"```json\n{eval_text}\n```",
            "",
            "### Fusion Result",
            f"```json\n{fusion_text}\n```",
            "",
            "### Predictor Input (Evidence Snapshot)",
            f"```json\n{predictor_input_text}\n```",
            "",
            "### Data Overview (Coverage)",
            f"```json\n{overview_text}\n```",
            "",
            "### Context Fill Report (RAG Backfill)",
            f"```json\n{fill_text}\n```",
            "",
            "### Execution Summary",
            f"```json\n{exec_text}\n```",
            "",
            "Now produce the final Markdown report. No JSON, no code fences in the output.",
        ])

