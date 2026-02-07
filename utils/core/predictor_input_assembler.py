"""Ordered, no-drop predictor input assembly with chunking support."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import tiktoken

from ..token_packer import count_tokens
from .multimodal_coverage import feature_key_set


@dataclass
class PredictorSection:
    """A single ordered predictor context section."""

    name: str
    priority: int
    text: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    feature_keys: List[str] = field(default_factory=list)


class PredictorInputAssembler:
    """
    Builds predictor sections in fixed priority order and packs them into chunks.

    No-drop policy: sections are never discarded; oversized sections are split into
    multiple slice sections.
    """

    def __init__(self, max_chunk_tokens: int = 12000, model_hint: str = "gpt-5"):
        self.max_chunk_tokens = int(max_chunk_tokens)
        self.model_hint = model_hint
        try:
            self._encoder = tiktoken.encoding_for_model(model_hint)
        except Exception:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    def build_sections(
        self,
        *,
        executor_output: Dict[str, Any],
        predictor_input: Dict[str, Any],
        coverage_ledger: Dict[str, Any],
    ) -> List[PredictorSection]:
        step_outputs = executor_output.get("step_outputs", {}) or {}

        def _as_text(obj: Any) -> str:
            if obj is None:
                return "Not provided"
            if isinstance(obj, str):
                return obj
            try:
                return json.dumps(obj, indent=2, default=str)
            except Exception:
                return str(obj)

        def _iter_step_rows() -> List[tuple[str, str, Dict[str, Any]]]:
            rows: List[tuple[str, str, Dict[str, Any]]] = []
            for sid, out in sorted(
                step_outputs.items(),
                key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]),
            ):
                if not isinstance(out, dict):
                    continue
                tool_name = str(out.get("tool_name") or (out.get("_step_meta") or {}).get("tool_name") or "")
                rows.append((str(sid), tool_name, out))
            return rows

        step_rows = _iter_step_rows()

        def _tool_group_text(names: Iterable[str], consumed: Optional[Set[str]] = None) -> str:
            allowed = {str(n) for n in names}
            rows = []
            for sid, tool_name, out in step_rows:
                if tool_name not in allowed:
                    continue
                if consumed is not None:
                    consumed.add(f"{sid}:{tool_name}")
                rows.append(f"## Step {sid} ({tool_name})\n{_as_text(out)}")
            return "\n\n".join(rows) if rows else "Not provided"

        def _remaining_tools_text(consumed: Set[str]) -> str:
            rows = []
            for sid, tool_name, out in step_rows:
                mark = f"{sid}:{tool_name}"
                if mark in consumed:
                    continue
                rows.append(f"## Step {sid} ({tool_name or 'UnknownTool'})\n{_as_text(out)}")
            return "\n\n".join(rows) if rows else "Not provided"

        non_num = predictor_input.get("non_numerical_data_raw", executor_output.get("non_numerical_data", ""))
        dev = predictor_input.get("hierarchical_deviation_raw", executor_output.get("hierarchical_deviation", {}))
        overview = executor_output.get("data_overview", {})

        consumed_tools: Set[str] = set()

        phenotype = _tool_group_text({"PhenotypeRepresentation"}, consumed_tools)
        feat_syn = _tool_group_text({"FeatureSynthesizer"}, consumed_tools)
        differential = _tool_group_text({"DifferentialDiagnosis"}, consumed_tools)
        clinical_rank = _tool_group_text({"ClinicalRelevanceRanker"}, consumed_tools)
        multimodal_narr = _tool_group_text({"MultimodalNarrativeCreator"}, consumed_tools)
        unimodal = _tool_group_text({"UnimodalCompressor"}, consumed_tools)
        remaining_tools = _remaining_tools_text(consumed_tools)

        unprocessed_raw = predictor_input.get("multimodal_unprocessed_raw") or {}
        processed_raw = predictor_input.get("multimodal_processed_raw_low_priority") or {}
        rag_fill = (predictor_input.get("context_fill_report") or {}).get("top_added") or []
        context_fill = predictor_input.get("context_fill_report") or {}
        payload_estimate = context_fill.get("predictor_payload_estimate") or {}
        aux = {
            "mode": predictor_input.get("mode"),
            "step_output_count": len(step_rows),
            "coverage_summary": (coverage_ledger or {}).get("summary") or {},
            "context_fill_summary": {
                "processed_raw_full_included": context_fill.get("processed_raw_full_included"),
                "rag_added_count": len(context_fill.get("top_added") or []),
                "predictor_payload_estimate": {
                    "baseline_tokens": payload_estimate.get("baseline_tokens"),
                    "final_tokens": payload_estimate.get("final_tokens"),
                    "single_chunk_limit": payload_estimate.get("single_chunk_limit"),
                    "threshold": payload_estimate.get("threshold"),
                },
            },
        }

        unprocessed_keys = sorted(list(feature_key_set(unprocessed_raw)))
        processed_keys = sorted(list(feature_key_set(processed_raw)))

        sections: List[PredictorSection] = [
            PredictorSection(
                name="non_numerical_data_raw",
                priority=1,
                text=_as_text(non_num),
                provenance={"source": "non_numerical_data.txt"},
                feature_keys=[],
            ),
            PredictorSection(
                name="hierarchical_deviation_raw",
                priority=2,
                text=_as_text(dev),
                provenance={"source": "hierarchical_deviation_map.json"},
                feature_keys=[],
            ),
            PredictorSection(
                name="data_overview",
                priority=3,
                text=_as_text(overview),
                provenance={"source": "data_overview.json"},
                feature_keys=[],
            ),
            PredictorSection("phenotype_representation", 4, phenotype, {"tools": ["PhenotypeRepresentation"]}, []),
            PredictorSection("feature_synthesizer", 5, feat_syn, {"tools": ["FeatureSynthesizer"]}, []),
            PredictorSection("differential_diagnosis", 6, differential, {"tools": ["DifferentialDiagnosis"]}, []),
            PredictorSection("clinical_relevance_ranker", 7, clinical_rank, {"tools": ["ClinicalRelevanceRanker"]}, []),
            PredictorSection(
                name="unprocessed_multimodal_raw",
                priority=8,
                text=_as_text(unprocessed_raw),
                provenance={"source": "multimodal_data.json", "coverage": "unprocessed_subtrees"},
                feature_keys=unprocessed_keys,
            ),
            PredictorSection("multimodal_narrative", 9, multimodal_narr, {"tools": ["MultimodalNarrativeCreator"]}, []),
            PredictorSection("unimodal_outputs", 10, unimodal, {"tools": ["UnimodalCompressor"]}, []),
            PredictorSection("remaining_tool_outputs", 11, remaining_tools, {"tools": ["other"]}, []),
            PredictorSection("auxiliary", 12, _as_text(aux), {"source": "aux"}, []),
            PredictorSection(
                "processed_multimodal_raw_low_priority",
                12,
                _as_text(processed_raw),
                {"source": "processed_raw_low_priority"},
                processed_keys,
            ),
            PredictorSection("rag_fill", 13, _as_text(rag_fill), {"source": "rag_fill_low_priority"}, []),
        ]

        forced_raw = coverage_ledger.get("forced_raw_features") or []
        if forced_raw:
            sections.insert(
                8,
                PredictorSection(
                    name="forced_raw_missing_features",
                    priority=8,
                    text=_as_text({"forced_raw_features": forced_raw}),
                    provenance={"reason": "coverage_invariant"},
                    feature_keys=list(forced_raw),
                )
            )

        return self._explode_oversized_sections(sections)

    def _explode_oversized_sections(self, sections: Sequence[PredictorSection]) -> List[PredictorSection]:
        """Split oversized sections into sequential slices without dropping data."""
        out: List[PredictorSection] = []
        for sec in sections:
            tok = count_tokens(sec.text, model_hint=self.model_hint)
            if tok <= max(128, self.max_chunk_tokens - 1200):
                out.append(sec)
                continue

            # Split exactly on token boundaries to avoid text loss.
            slice_budget = max(512, self.max_chunk_tokens - 1500)
            sec_tokens = self._encoder.encode(sec.text or "")
            parts_count = max(1, (len(sec_tokens) + slice_budget - 1) // slice_budget)
            if parts_count == 1:
                out.append(
                    PredictorSection(
                        name=sec.name,
                        priority=sec.priority,
                        text=sec.text,
                        provenance=dict(sec.provenance),
                        feature_keys=list(sec.feature_keys),
                    )
                )
                continue

            if sec.feature_keys:
                keys_per_part = max(1, (len(sec.feature_keys) + parts_count - 1) // parts_count)
            else:
                keys_per_part = 0

            for part_idx, start in enumerate(range(0, len(sec_tokens), slice_budget), 1):
                end = min(start + slice_budget, len(sec_tokens))
                part_text = self._encoder.decode(sec_tokens[start:end])
                key_slice: List[str] = []
                if keys_per_part:
                    ks = (part_idx - 1) * keys_per_part
                    ke = min(len(sec.feature_keys), part_idx * keys_per_part)
                    key_slice = list(sec.feature_keys[ks:ke])

                out.append(
                    PredictorSection(
                        name=f"{sec.name}#part{part_idx}",
                        priority=sec.priority,
                        text=part_text,
                        provenance={**sec.provenance, "part": part_idx, "parts_total": parts_count},
                        feature_keys=key_slice,
                    )
                )

        return out

    def build_chunks(self, sections: Sequence[PredictorSection]) -> List[List[PredictorSection]]:
        """Pack ordered sections into <=max_chunk_tokens chunks without reordering."""
        def _pack(items: Sequence[PredictorSection]) -> List[List[PredictorSection]]:
            packed: List[List[PredictorSection]] = []
            current: List[PredictorSection] = []
            current_tokens = 0

            for sec in items:
                wrapped = self._section_to_text(sec)
                wrapped_tokens = count_tokens(wrapped, model_hint=self.model_hint)

                if current and (current_tokens + wrapped_tokens > self.max_chunk_tokens):
                    packed.append(current)
                    current = []
                    current_tokens = 0

                current.append(sec)
                current_tokens += wrapped_tokens

            if current:
                packed.append(current)

            return packed

        chunks = _pack(sections)
        if len(chunks) > 1:
            filtered = [s for s in sections if s.name != "processed_multimodal_raw_low_priority"]
            if len(filtered) != len(sections):
                return _pack(filtered)
        return chunks

    def _section_to_text(self, sec: PredictorSection) -> str:
        head = {
            "section": sec.name,
            "priority": sec.priority,
            "provenance": sec.provenance,
            "feature_key_count": len(sec.feature_keys),
            "feature_keys_sample": sec.feature_keys[:50],
        }
        return f"### SECTION_META\n{json.dumps(head, default=str)}\n### SECTION_TEXT\n{sec.text}"

    def chunk_to_text(self, chunk: Sequence[PredictorSection], chunk_index: int, chunk_total: int) -> str:
        body = "\n\n".join(self._section_to_text(s) for s in chunk)
        return (
            f"## PREDICTOR_CONTEXT_CHUNK {chunk_index}/{chunk_total}\n"
            f"This chunk is part of a no-loss multi-chunk analysis.\n\n{body}"
        )
