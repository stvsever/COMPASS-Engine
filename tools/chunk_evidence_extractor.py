"""
COMPASS Chunk Evidence Extractor Tool

Summarizes a single predictor chunk into structured evidence.
"""

from typing import Dict, Any, Optional

from .base_tool import BaseTool


class ChunkEvidenceExtractor(BaseTool):
    """
    Extracts chunk-level evidence for CASE/CONTROL reasoning.
    """

    TOOL_NAME = "ChunkEvidenceExtractor"
    PROMPT_FILE = "chunk_evidence_extractor.txt"
    TOOL_POLICY_SCOPE = "local"
    TOOL_MAX_TOKENS = 2048
    TOOL_TEMPERATURE = 0.0
    TOOL_MAX_RETRIES = 2

    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        required = ["chunk_text", "target_condition", "control_condition", "chunk_index", "chunk_total"]
        for key in required:
            if key not in input_data:
                return f"Missing required input: {key}"
        return None

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        chunk_text = input_data.get("chunk_text", "")
        target = input_data.get("target_condition", "")
        control = input_data.get("control_condition", "")
        chunk_index = input_data.get("chunk_index", 0)
        chunk_total = input_data.get("chunk_total", 0)
        hinted_keys = input_data.get("hinted_feature_keys") or []

        return "\n".join([
            f"Target condition: {target}",
            f"Control condition: {control}",
            f"Chunk: {chunk_index}/{chunk_total}",
            "",
            "Chunk feature key hints (may be partial):",
            str(hinted_keys),
            "",
            "Chunk content:",
            f"```text\n{chunk_text}\n```",
            "",
            "OUTPUT CONTRACT:",
            "- Return only one JSON object matching the schema.",
            "- Do not output analysis or <think> blocks.",
            "- Keep key_findings concise (max 8 items).",
            "- Keep for_case and for_control concise (max 6 items each).",
        ])
