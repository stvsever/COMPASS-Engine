"""
COMPASS Clinical Relevance Ranker Tool

Ranks features by clinical relevance for the target condition.
"""

import json
import re
from typing import Dict, Any, Optional, List

from .base_tool import BaseTool


class ClinicalRelevanceRanker(BaseTool):
    """
    Ranks features by clinical relevance for prediction.
    
    Uses medical knowledge to prioritize features that are
    most informative for target phenotype/phenotype comparator disorders.
    """
    
    TOOL_NAME = "ClinicalRelevanceRanker"
    PROMPT_FILE = "clinical_ranker.txt"
    TOOL_EXPECTED_KEYS = [
        "clinical_relevance_overview",
        "case_control_discrimination",
        "predictor_guidance",
    ]
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate that required inputs are present."""
        if "target_condition" not in input_data:
            return "Missing target_condition"
        
        return None
    
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the clinical ranking prompt."""
        target = input_data.get("target_condition", "target phenotype")
        control = input_data.get("control_condition", "")
        
        # Get features from dependency outputs or hierarchical deviation
        dep_outputs = input_data.get("dependency_outputs", {})
        hierarchical_deviation = input_data.get("hierarchical_deviation", {})
        synthesis_context = self._extract_synthesis_context(dep_outputs)
        
        features = self._collect_features(dep_outputs, hierarchical_deviation)
        
        prompt_parts = [
            f"## TARGET CONDITION: {target}",
            f"## CONTROL CONDITION: {control}",
            
            f"\n## FEATURES TO RANK",
            f"```json\n{json.dumps(features[:30], indent=2)}\n```",

            f"\n## FEATURE SYNTHESIS CONTEXT",
            synthesis_context if synthesis_context else "No prior synthesis narrative provided",
            
            f"\n## CURRENT ABNORMALITY PROFILE",
            self._get_abnormality_profile(hierarchical_deviation),
            
            "\n## CLINICAL RANKING GUIDELINES",
            self._get_clinical_guidelines(target),
            
            "\n## TASK",
            f"Assess clinical relevance signal for {target} prediction.",
            "Consider established biomarkers and clinical evidence.",
            "Focus on task-output discrimination quality (binary: case-vs-control), uncertainty, and practical predictor guidance.",
            "Return detailed free-text synthesis only; do not return per-feature ranked lists."
        ]
        
        return "\n".join(prompt_parts)

    def _extract_synthesis_context(self, dep_outputs: Dict[str, Any]) -> str:
        """Extract narrative context from FeatureSynthesizer outputs."""
        contexts: List[str] = []
        for _, output in dep_outputs.items():
            if not isinstance(output, dict):
                continue
            for key in (
                "feature_synthesis_overview",
                "domain_signal_overview",
                "hierarchy_signal_overview",
                "predictor_attention_guidance",
            ):
                text = str(output.get(key) or "").strip()
                if text:
                    contexts.append(text)
        if not contexts:
            return ""
        # Keep prompt compact while preserving the strongest context.
        return "\n".join(f"- {c}" for c in contexts[:4])
    
    def _collect_features(
        self,
        dep_outputs: Dict[str, Any],
        hierarchical_deviation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Collect features from various sources."""
        features = []
        
        # From dependency outputs (FeatureSynthesizer + others)
        for _, output in dep_outputs.items():
            if isinstance(output, dict):
                # New schema used by FeatureSynthesizer
                if isinstance(output.get("top_features"), list):
                    for item in output["top_features"]:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("feature_name") or item.get("feature") or "").strip()
                        if not name:
                            continue
                        features.append(
                            {
                                "feature": name,
                                "z_score": item.get("z_score"),
                                "importance_score": item.get("importance_score"),
                                "domain": item.get("domain", "UNKNOWN"),
                                "clinical_interpretation": item.get("clinical_interpretation", ""),
                            }
                        )

                # Legacy schemas
                if "top_10_features" in output:
                    features.extend(output["top_10_features"])
                if "feature_rankings" in output:
                    rankings = output["feature_rankings"]
                    if "top_10_features" in rankings:
                        features.extend(rankings["top_10_features"])
                if isinstance(output.get("ranked_features"), list):
                    for item in output["ranked_features"]:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("feature") or item.get("feature_name") or "").strip()
                        if not name:
                            continue
                        features.append(
                            {
                                "feature": name,
                                "z_score": item.get("z_score"),
                                "domain": item.get("domain", "UNKNOWN"),
                                "clinical_interpretation": item.get("rationale", ""),
                            }
                        )
        
        # From hierarchical deviation if no features yet
        if not features and hierarchical_deviation:
            features = self._extract_from_deviation(hierarchical_deviation)
        
        # Deduplicate by normalized feature label, keep first occurrence.
        seen = set()
        deduped = []
        for item in features:
            if not isinstance(item, dict):
                continue
            name = str(item.get("feature") or item.get("feature_name") or "").strip()
            if not name:
                continue
            key = re.sub(r"\s+", " ", name.lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        return deduped[:30]
    
    def _extract_from_deviation(
        self,
        deviation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract feature list from deviation structure."""
        features = []
        
        def traverse(node: Dict[str, Any]):
            if node.get("z_score") is not None and abs(node.get("z_score", 0)) > 1.0:
                features.append({
                    "feature": node.get("node_name", ""),
                    "z_score": node.get("z_score"),
                    "domain": self._get_parent_domain(node)
                })
            
            for child in node.get("children", []):
                traverse(child)
        
        if "root" in deviation:
            traverse(deviation["root"])
        
        return sorted(features, key=lambda f: abs(f.get("z_score", 0)), reverse=True)
    
    def _get_parent_domain(self, node: Dict[str, Any]) -> str:
        """Try to determine parent domain from node path or name."""
        name = node.get("node_name", "").upper()
        for domain in ["BRAIN", "GENOMICS", "COGNITION", "BIOLOGICAL_ASSAY", "DEMOGRAPHICS", "LIFESTYLE"]:
            if domain in name:
                return domain
        return "UNKNOWN"
    
    def _get_abnormality_profile(self, deviation: Dict[str, Any]) -> str:
        """Summarize current abnormalities."""
        if not deviation:
            return "No abnormality data"
        
        if "domain_summaries" in deviation:
            lines = []
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
                        lines.append(f"- {domain}: {severity or 'UNKNOWN'} ({suffix})")
                    else:
                        lines.append(f"- {domain}: {severity or 'UNKNOWN'}")
            return "\n".join(lines)
        
        return "Deviation structure available but not summarized"

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
    
    def _get_clinical_guidelines(self, target: str) -> str:
        """Get condition-specific clinical guidelines."""
        if target == "target phenotype":
            return """
For target phenotype conditions, prioritize:
- HIGH: Limbic volumes, prefrontal cortex, stress markers, cognitive tests
- MEDIUM: Global brain measures, inflammatory markers
- LOWER: Motor cortex, non-specific markers
"""
        else:
            return """
For phenotype comparator conditions, prioritize:
- HIGH: Global atrophy, white matter lesions, genetic variants, CSF markers
- MEDIUM: Cognitive performance, metabolic factors
- LOWER: Mood measures, social factors
"""

    def _process_output(
        self,
        output_data: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize to narrative-only clinical relevance output."""
        if not isinstance(output_data, dict):
            output_data = {}

        def _clean_text(value: Any) -> str:
            return " ".join(str(value or "").split()).strip()

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

        legacy_ranked = output_data.get("ranked_features")
        if not isinstance(legacy_ranked, list):
            legacy_ranked = []
        legacy_priorities = output_data.get("top_clinical_priorities")
        if not isinstance(legacy_priorities, list):
            legacy_priorities = []

        top_domains: List[str] = []
        evidence_tags: List[str] = []
        for item in legacy_ranked[:12]:
            if not isinstance(item, dict):
                continue
            domain = _clean_text(item.get("domain"))
            if domain:
                top_domains.append(domain)
            relevance = _clean_text(item.get("clinical_relevance"))
            if relevance:
                evidence_tags.append(relevance.upper())

        domain_summary = ", ".join(dict.fromkeys(top_domains))
        relevance_summary = ", ".join(dict.fromkeys(evidence_tags))
        priorities_text = "; ".join(
            _clean_text(p) for p in legacy_priorities if _clean_text(p) and not _looks_placeholder(_clean_text(p))
        )

        clinical_relevance_overview = _first_valid(
            output_data.get("clinical_relevance_overview"),
            output_data.get("clinical_summary"),
            output_data.get("summary"),
            output_data.get("narrative"),
        )
        if not clinical_relevance_overview:
            if domain_summary or priorities_text:
                clinical_relevance_overview = (
                    "Clinical relevance signal is concentrated in domains "
                    + (domain_summary if domain_summary else "identified by upstream synthesis")
                    + ". "
                    + (
                        f"Key clinical priorities indicate: {priorities_text}. "
                        if priorities_text
                        else ""
                    )
                    + "Use this signal as weighted context rather than deterministic evidence from any single marker."
                )
            else:
                clinical_relevance_overview = (
                    "Clinical relevance cannot be narrowed to a single domain; interpret findings through multimodal consistency, symptom context, and target-specific plausibility."
                )

        case_control_discrimination = _first_valid(
            output_data.get("case_control_discrimination"),
            output_data.get("discrimination_summary"),
            output_data.get("control_separation_rationale"),
        )
        if not case_control_discrimination:
            if relevance_summary:
                case_control_discrimination = (
                    f"Evidence strength profile ({relevance_summary}) supports weighting convergent clinically plausible signals over isolated biomarker outliers when separating plausible task outputs."
                )
            else:
                case_control_discrimination = (
                    "Differentiate plausible outputs by requiring coherent target-aligned evidence across clinical narrative, deviations, and non-numerical context; in binary mode, favor CONTROL under sparse or contradictory evidence."
                )

        predictor_guidance = _first_valid(
            output_data.get("predictor_guidance"),
            output_data.get("prediction_guidance"),
            output_data.get("actionable_guidance"),
        )
        if not predictor_guidance:
            predictor_guidance = (
                "Prioritize clinically coherent multi-domain patterns first, down-weight non-specific abnormalities, and explicitly test whether observed evidence is more consistent with control-pathology mechanisms than target syndrome mechanisms."
            )

        uncertainty_and_gaps = _first_valid(
            output_data.get("uncertainty_and_gaps"),
            output_data.get("evidence_gaps"),
            output_data.get("missing_key_features_summary"),
        )
        if not uncertainty_and_gaps:
            uncertainty_and_gaps = (
                "Residual uncertainty should be resolved using symptom-level evidence, temporal progression, and confirmatory cross-domain consistency checks."
            )

        return {
            "ranking_id": _clean_text(output_data.get("ranking_id")),
            "target_condition": _clean_text(
                output_data.get("target_condition") or input_data.get("target_condition")
            ),
            "clinical_relevance_overview": clinical_relevance_overview,
            "case_control_discrimination": case_control_discrimination,
            "predictor_guidance": predictor_guidance,
            "uncertainty_and_gaps": uncertainty_and_gaps,
        }
