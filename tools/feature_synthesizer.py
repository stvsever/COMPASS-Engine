"""
COMPASS Feature Synthesizer Tool

Synthesizes feature importance from hierarchical data structures.
"""

import json
from collections import Counter
from typing import Dict, Any, Optional, List

from .base_tool import BaseTool


class FeatureSynthesizer(BaseTool):
    """
    Synthesizes feature importance from hierarchical deviation data.
    
    Identifies most discriminative features and ranks by
    predictive power for the target condition.
    """
    
    TOOL_NAME = "FeatureSynthesizer"
    PROMPT_FILE = "feature_synthesizer.txt"
    TOOL_EXPECTED_KEYS = [
        "feature_synthesis_overview",
        "domain_signal_overview",
        "hierarchy_signal_overview",
        "predictor_attention_guidance",
    ]
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate that required inputs are present."""
        if not input_data.get("hierarchical_deviation"):
            return "Missing hierarchical_deviation data"
        
        if "target_condition" not in input_data:
            return "Missing target_condition"
        
        return None
    
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the feature synthesis prompt."""
        target = input_data.get("target_condition", "target phenotype")
        control = input_data.get("control_condition", "")
        hierarchical_deviation = input_data.get("hierarchical_deviation", {})
        domains = input_data.get("input_domains", [])
        
        # Extract features with z-scores
        features = self._extract_features(hierarchical_deviation)
        
        prompt_parts = [
            f"## TARGET CONDITION: {target}",
            f"## CONTROL CONDITION: {control}",
            f"## DOMAINS WITH DATA: {', '.join(domains) if domains else 'All available'}",
            
            f"\n## FEATURES WITH DEVIATIONS",
            f"Total features extracted: {len(features)}",
            f"```json\n{json.dumps(features[:50], indent=2)}\n```",
            
            f"\n## HIERARCHICAL STRUCTURE",
            self._describe_hierarchy(hierarchical_deviation),
            
            "\n## TASK",
            "Synthesize feature importance from the hierarchical structure.",
            "Identify the most discriminative features for the target condition.",
            "Rank features by predictive power.",
            "IMPORTANT: look at the actual data_overview what data is present! Do not mention (hierarchical) features that are NOT inside the data.",
            "Group by domain and provide aggregate assessments."
        ]
        
        return "\n".join(prompt_parts)

    def _process_output(
        self,
        output_data: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize to narrative-only synthesis output for predictor guidance."""
        if not isinstance(output_data, dict):
            output_data = {}

        def _clean_text(value: Any) -> str:
            text = str(value or "").strip()
            return " ".join(text.split())

        def _looks_placeholder(text: str) -> bool:
            low = text.lower()
            if not low:
                return True
            markers = (
                "no feature provided",
                "feature not provided",
                "not provided",
                "n/a",
                "none",
                "null",
                "unknown",
                "placeholder",
            )
            return any(marker in low for marker in markers)

        def _first_valid(*values: Any) -> str:
            for value in values:
                text = _clean_text(value)
                if text and not _looks_placeholder(text):
                    return text
            return ""

        def _to_float(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                return float(value)
            except Exception:
                return None

        # Legacy compatibility: derive narrative from old list-heavy payloads.
        legacy_features = output_data.get("top_features")
        if not isinstance(legacy_features, list):
            legacy_features = []

        domain_counter: Counter[str] = Counter()
        level_counter: Counter[str] = Counter()
        for item in legacy_features:
            if not isinstance(item, dict):
                continue
            dom = _clean_text(item.get("domain") or "UNKNOWN") or "UNKNOWN"
            level = _clean_text(item.get("hierarchy_level") or item.get("level") or "leaf") or "leaf"
            domain_counter[dom] += 1
            level_counter[level.lower()] += 1

        legacy_domain_importance = output_data.get("domain_importance")
        if not isinstance(legacy_domain_importance, list):
            legacy_domain_importance = []

        prioritized_domains: List[str] = []
        for item in legacy_domain_importance:
            if not isinstance(item, dict):
                continue
            dom = _clean_text(item.get("domain"))
            if dom:
                prioritized_domains.append(dom)
        if not prioritized_domains:
            prioritized_domains = [dom for dom, _ in domain_counter.most_common(4)]

        root_pattern = ""
        hierarchy_summary = output_data.get("hierarchy_summary")
        if isinstance(hierarchy_summary, dict):
            root_pattern = _clean_text(hierarchy_summary.get("root_pattern"))

        hierarchy_map = output_data.get("hierarchical_attention_map")
        if not isinstance(hierarchy_map, dict):
            hierarchy_map = {}

        predictor_guidance = ""
        attention_directives = output_data.get("attention_directives")
        if isinstance(attention_directives, dict):
            predictor_guidance = _clean_text(attention_directives.get("predictor_guidance"))

        feature_synthesis_overview = _first_valid(
            output_data.get("feature_synthesis_overview"),
            output_data.get("predictive_signal_overview"),
            output_data.get("summary"),
            output_data.get("narrative"),
            output_data.get("clinical_summary"),
        )
        if not feature_synthesis_overview:
            if prioritized_domains:
                feature_synthesis_overview = (
                    "Predictive signal is concentrated in "
                    + ", ".join(prioritized_domains[:4])
                    + ". Distinguish plausible task outputs using cross-domain coherence, "
                      "consistency of abnormal patterns, and whether deviations align with target-specific mechanisms."
                )
            else:
                feature_synthesis_overview = (
                    "Predictive signal is diffuse with no dominant single-feature driver; "
                    "classification should rely on multimodal pattern consistency and symptom-context alignment."
                )

        domain_signal_overview = _first_valid(
            output_data.get("domain_signal_overview"),
            output_data.get("domain_attention_overview"),
            output_data.get("domain_priority_overview"),
        )
        if not domain_signal_overview:
            if prioritized_domains:
                domain_signal_overview = (
                    "Domain-level weighting favors "
                    + ", ".join(prioritized_domains[:5])
                    + ". Prioritize domains where multiple related deviations converge and deprioritize isolated single-domain anomalies."
                )
            else:
                domain_signal_overview = (
                    "No stable domain dominance detected; keep balanced attention across available domains and update weights using downstream clinical relevance synthesis."
                )

        hierarchy_signal_overview = _first_valid(
            output_data.get("hierarchy_signal_overview"),
            output_data.get("hierarchy_attention_overview"),
            root_pattern,
        )
        if not hierarchy_signal_overview:
            root_nodes = len(hierarchy_map.get("root") or [])
            subsystem_nodes = len(hierarchy_map.get("subsystem") or [])
            leaf_nodes = len(hierarchy_map.get("leaf") or [])
            if any((root_nodes, subsystem_nodes, leaf_nodes)):
                hierarchy_signal_overview = (
                    f"Hierarchy signal spans root={root_nodes}, subsystem={subsystem_nodes}, leaf={leaf_nodes} nodes, "
                    "with interpretation weighted toward cross-level agreement rather than isolated leaf anomalies."
                )
            elif level_counter:
                hierarchy_signal_overview = (
                    "Hierarchy evidence is weighted toward "
                    + ", ".join(f"{k}:{v}" for k, v in level_counter.most_common(3))
                    + "; prioritize patterns that remain coherent across hierarchy levels."
                )
            else:
                hierarchy_signal_overview = (
                    "Hierarchy-level emphasis cannot be estimated confidently from current output; "
                    "treat the signal as low-structure and require corroboration from multimodal narratives."
                )

        predictor_attention_guidance = _first_valid(
            output_data.get("predictor_attention_guidance"),
            output_data.get("predictor_guidance"),
            predictor_guidance,
        )
        if not predictor_attention_guidance:
            predictor_attention_guidance = (
                "Weight the strongest convergent domains first, validate against phenotype and non-numerical clinical context, "
                "and in binary mode prefer CONTROL when deviations are weak, scattered, or inconsistent with target-specific presentation."
            )

        # Accept legacy numeric hints in free text only; never emit explicit feature arrays.
        score_hint = _to_float(output_data.get("overall_predictive_signal"))
        if score_hint is not None:
            score_hint = max(0.0, min(1.0, score_hint))
            feature_synthesis_overview = (
                f"{feature_synthesis_overview} Estimated overall predictive signal strength: {score_hint:.2f}."
            )

        return {
            "synthesis_id": _clean_text(output_data.get("synthesis_id")),
            "target_condition": _clean_text(
                output_data.get("target_condition") or input_data.get("target_condition")
            ),
            "feature_synthesis_overview": feature_synthesis_overview,
            "domain_signal_overview": domain_signal_overview,
            "hierarchy_signal_overview": hierarchy_signal_overview,
            "predictor_attention_guidance": predictor_attention_guidance,
        }
    
    def _extract_features(self, deviation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all features with their z-scores."""
        features = []
        
        def traverse(node: Dict[str, Any], path: List[str]):
            node_name = node.get("node_name") or node.get("name") or "unknown"
            current_path = path + [node_name]
            
            # Check for direct z_score (leaf) or aggregated score (node)
            z_score = node.get("z_score")
            stats = node.get("_stats", {})
            mean_abs = stats.get("mean_abs_score")
            
            # If we have a score (either direct or aggregated)
            if z_score is not None or mean_abs is not None:
                # Use z_score if available, otherwise use mean_abs (with sign inference if possible, else absolute)
                # Note: mean_abs is always positive, so we default to HIGH direction if we don't know
                score_val = z_score if z_score is not None else mean_abs
                
                features.append({
                    "feature_id": node.get("node_id", ""),
                    "name": node.get("node_name", node.get("name", current_path[-1])),
                    "path": " > ".join(current_path),
                    "z_score": score_val,
                    "is_aggregated": z_score is None,
                    "direction": "HIGH" if (z_score and z_score > 0) or (mean_abs and mean_abs > 0) else "LOW",
                    "severity": self._classify_severity(score_val)
                })
            
            # Traverse explicit children list (tree format)
            children_list = node.get("children")
            if isinstance(children_list, list):
                for child in children_list:
                    if isinstance(child, dict):
                        traverse(child, current_path)

            # Traverse nested dict children (UKB nested format)
            for key, child in node.items():
                if key in {"_stats", "children"}:
                    continue
                if isinstance(child, dict):
                    if "node_name" not in child and "name" not in child:
                        child = dict(child)
                        child["node_name"] = key
                    traverse(child, current_path)
        
        if "root" in deviation:
            traverse(deviation["root"], [])
        else:
            # Handle case where top-level keys are domains directly
            for key, val in deviation.items():
                if isinstance(val, dict) and key != "_stats":
                    # Add node_name if missing
                    if "node_name" not in val:
                        val["node_name"] = key
                    traverse(val, [])
        
        # Sort by absolute z-score
        features.sort(key=lambda f: abs(f.get("z_score", 0)), reverse=True)
        
        return features
    
    def _classify_severity(self, z_score: Optional[float]) -> str:
        """Classify severity based on z-score."""
        if z_score is None:
            return "NORMAL"
        
        abs_z = abs(z_score)
        if abs_z > 3.0:
            return "SEVERE"
        elif abs_z > 2.0:
            return "MODERATE"
        elif abs_z > 1.5:
            return "MILD"
        return "NORMAL"
    
    def _describe_hierarchy(self, deviation: Dict[str, Any]) -> str:
        """Create text description of hierarchy structure."""
        if not deviation:
            return "Hierarchy structure not available"
        
        if "root" in deviation:
            root = deviation["root"]
            domains = [c.get("node_name", "") for c in root.get("children", [])]
        else:
            # Extract top-level keys as domains
            domains = [k for k in deviation.keys() if k != "_stats"]
            
        return f"Hierarchy with {len(domains)} domain branches: {', '.join(domains)}"
