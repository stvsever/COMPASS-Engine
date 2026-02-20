"""
COMPASS Hypothesis Generator Tool

Generates biomedical hypotheses explaining observed deviations.
"""

import json
import re
from typing import Dict, Any, Optional

from .base_tool import BaseTool


class HypothesisGenerator(BaseTool):
    """
    Generates plausible biomedical hypotheses for abnormalities.
    
    Links deviations to target phenotype/phenotype comparator conditions
    through established pathophysiological mechanisms.
    """
    
    TOOL_NAME = "HypothesisGenerator"
    PROMPT_FILE = "hypothesis_generator.txt"
    TOOL_EXPECTED_KEYS = [
        "primary_hypothesis",
        "alternative_hypotheses",
        "differential_considerations",
    ]
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate that required inputs are present."""
        if "target_condition" not in input_data:
            return "Missing target_condition"
        
        if not input_data.get("hierarchical_deviation"):
            return "Missing hierarchical_deviation data"
        
        return None
    
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the hypothesis generation prompt."""
        target = input_data.get("target_condition", "target phenotype")
        control = input_data.get("control_condition", "")
        
        # Get abnormality data from dependency outputs or hierarchical deviation
        dep_outputs = input_data.get("dependency_outputs", {})
        hierarchical_deviation = input_data.get("hierarchical_deviation", {})
        
        # Collect abnormalities from various sources
        abnormalities = self._collect_abnormalities(dep_outputs, hierarchical_deviation)
        
        # Get patient context
        participant_id = input_data.get("participant_id", "unknown")
        
        prompt_parts = [
            f"## TARGET CONDITION: {target}",
            f"## CONTROL CONDITION: {control}",
            f"## PARTICIPANT: {participant_id}",
            
            f"\n## DETECTED ABNORMALITIES",
            json.dumps(abnormalities, indent=2)[:3000],
            
            f"\n## HIERARCHICAL DEVIATION PROFILE",
            self._format_deviation_summary(hierarchical_deviation),
            
            "\n## TASK",
            f"Generate biomedical hypotheses explaining these abnormalities.",
            f"Link to {target} disorder mechanisms.",
            "Provide primary and alternative hypotheses.",
            "Include differential considerations."
        ]
        
        return "\n".join(prompt_parts)
    
    def _collect_abnormalities(
        self,
        dep_outputs: Dict[str, Any],
        hierarchical_deviation: Dict[str, Any]
    ) -> list:
        """Collect all abnormalities from various sources."""
        abnormalities = []
        seen_labels = set()

        def _push(entry: Dict[str, Any]) -> None:
            label = str(
                entry.get("feature")
                or entry.get("pattern_name")
                or entry.get("description")
                or entry.get("domain")
                or ""
            ).strip()
            key = re.sub(r"\s+", " ", label.lower())
            if key and key not in seen_labels:
                seen_labels.add(key)
                abnormalities.append(entry)
        
        # From dependency outputs
        for _, output in dep_outputs.items():
            if isinstance(output, dict):
                if "key_abnormalities" in output:
                    for item in output["key_abnormalities"][:5]:
                        if isinstance(item, dict):
                            _push(item)
                        elif item:
                            _push({"description": str(item)})

                if isinstance(output.get("abnormality_patterns"), list):
                    for pattern in output["abnormality_patterns"][:5]:
                        if not isinstance(pattern, dict):
                            continue
                        _push(
                            {
                                "domain": output.get("domain") or output.get("base_domain") or "UNKNOWN",
                                "pattern_name": pattern.get("pattern_name"),
                                "severity": pattern.get("severity"),
                                "description": pattern.get("clinical_interpretation"),
                            }
                        )

                for key, tag in (
                    ("feature_synthesis_overview", "feature_synthesis"),
                    ("domain_signal_overview", "feature_synthesis"),
                    ("hierarchy_signal_overview", "feature_synthesis"),
                    ("clinical_relevance_overview", "clinical_relevance"),
                    ("case_control_discrimination", "clinical_relevance"),
                ):
                    text = str(output.get(key) or "").strip()
                    if text:
                        _push(
                            {
                                "domain": "CROSS_DOMAIN",
                                "feature": tag,
                                "description": text,
                            }
                        )

                if isinstance(output.get("top_features"), list):
                    for feat in output["top_features"][:5]:
                        if not isinstance(feat, dict):
                            continue
                        _push(
                            {
                                "domain": feat.get("domain", "UNKNOWN"),
                                "feature": feat.get("feature_name") or feat.get("feature"),
                                "importance_score": feat.get("importance_score"),
                                "description": feat.get("clinical_interpretation"),
                            }
                        )

                if isinstance(output.get("ranked_features"), list):
                    for feat in output["ranked_features"][:3]:
                        if not isinstance(feat, dict):
                            continue
                        _push(
                            {
                                "feature": feat.get("feature") or feat.get("feature_name"),
                                "description": feat.get("rationale"),
                                "clinical_relevance": feat.get("clinical_relevance"),
                            }
                        )
        
        # From domain summaries
        if "domain_summaries" in hierarchical_deviation:
            for domain, summary in hierarchical_deviation["domain_summaries"].items():
                if isinstance(summary, dict):
                    severity = summary.get("severity")
                    mean_abs = summary.get("mean_abs_score")
                    if not severity and mean_abs is not None:
                        severity = self._severity_from_mean(mean_abs)
                    if severity in ["SEVERE", "MODERATE"]:
                        _push({
                            "domain": domain,
                            "severity": severity,
                            "description": f"{domain} shows {severity} abnormality"
                        })
        
        return abnormalities[:15]  # Limit to top 15
    
    def _format_deviation_summary(self, deviation: Dict[str, Any]) -> str:
        """Format deviation for prompt."""
        if not deviation:
            return "No deviation data available"
        
        parts = []
        if "domain_summaries" in deviation:
            for domain, summary in deviation["domain_summaries"].items():
                if isinstance(summary, dict):
                    severity = summary.get("severity")
                    mean_abs = summary.get("mean_abs_score")
                    n_leaves = summary.get("n_leaves")
                    if not severity and mean_abs is not None:
                        severity = self._severity_from_mean(mean_abs)
                    if mean_abs is not None:
                        suffix = f"mean_abs={mean_abs:.3f}"
                        if n_leaves is not None:
                            suffix += f", n={n_leaves}"
                        parts.append(f"- {domain}: {severity or 'UNKNOWN'} ({suffix})")
                    else:
                        parts.append(f"- {domain}: {severity or 'UNKNOWN'}")
                else:
                    parts.append(f"- {domain}: {str(summary)[:100]}")
        
        return "\n".join(parts) if parts else "Deviation format not recognized"

    def _severity_from_mean(self, mean_abs: Optional[float]) -> str:
        """Infer severity from mean_abs_score (UKB format)."""
        if mean_abs is None:
            return "UNKNOWN"
        if mean_abs > 3.0:
            return "SEVERE"
        if mean_abs > 2.0:
            return "MODERATE"
        if mean_abs > 1.5:
            return "MILD"
        return "NORMAL"

    def _process_output(
        self,
        output_data: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize hypothesis payload and add concise summary for UI previewing."""
        if not isinstance(output_data, dict):
            return {}

        primary = output_data.get("primary_hypothesis")
        if isinstance(primary, dict):
            title = str(primary.get("title") or "").strip()
            mechanism = str(primary.get("mechanism") or "").strip()
            if title and mechanism:
                output_data["hypothesis_summary"] = f"{title}: {mechanism}"
            elif title:
                output_data["hypothesis_summary"] = title
            elif mechanism:
                output_data["hypothesis_summary"] = mechanism

        if "hypothesis_summary" not in output_data:
            alt = output_data.get("alternative_hypotheses") or []
            if isinstance(alt, list) and alt:
                first = alt[0]
                if isinstance(first, dict):
                    title = str(first.get("title") or "").strip()
                    mechanism = str(first.get("mechanism") or "").strip()
                    if title and mechanism:
                        output_data["hypothesis_summary"] = f"{title}: {mechanism}"
                    elif title:
                        output_data["hypothesis_summary"] = title

        return output_data
