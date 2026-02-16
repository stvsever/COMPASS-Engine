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
    # Hard cap this tool's output to 1K tokens (overrides global tool_out budget).
    TOOL_MAX_TOKENS = 1024 # would take too long otherwise
    TOOL_TEMPERATURE = 0.0
    TOOL_MAX_RETRIES = 2
    TOOL_EXPECTED_KEYS = [
        "summary",
        "for_case",
        "for_control",
        "uncertainty_factors",
        "key_findings",
        "cited_feature_keys",
    ]

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

    def _process_output(
        self,
        output_data: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        def _as_str_list(value: Any, max_items: int) -> list[str]:
            if not isinstance(value, list):
                return []
            out: list[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if not text:
                    continue
                out.append(text)
                if len(out) >= max_items:
                    break
            return out

        def _normalize_findings(value: Any, max_items: int = 8) -> list[Dict[str, str]]:
            if not isinstance(value, list):
                return []
            rows: list[Dict[str, str]] = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                finding = str(item.get("finding") or "").strip()
                if not finding:
                    continue
                rows.append(
                    {
                        "domain": str(item.get("domain") or "SYSTEM").strip() or "SYSTEM",
                        "finding": finding,
                        "direction": str(item.get("direction") or "NORMAL").strip() or "NORMAL",
                    }
                )
                if len(rows) >= max_items:
                    break
            return rows

        if not isinstance(output_data, dict):
            if isinstance(output_data, list):
                first_obj = next((item for item in output_data if isinstance(item, dict)), None)
                output_data = first_obj if isinstance(first_obj, dict) else {}
            else:
                output_data = {}

        summary = str(output_data.get("summary") or "").strip()
        if not summary:
            summary = "No explicit findings extracted from this chunk."

        hinted_keys = _as_str_list(input_data.get("hinted_feature_keys"), 120)
        cited = _as_str_list(output_data.get("cited_feature_keys"), 120)
        if cited:
            cited = sorted(set(cited) | set(hinted_keys))
        else:
            cited = hinted_keys

        return {
            "summary": summary,
            "for_case": _as_str_list(output_data.get("for_case"), 6),
            "for_control": _as_str_list(output_data.get("for_control"), 6),
            "uncertainty_factors": _as_str_list(output_data.get("uncertainty_factors"), 8),
            "key_findings": _normalize_findings(output_data.get("key_findings"), 8),
            "cited_feature_keys": cited,
        }
