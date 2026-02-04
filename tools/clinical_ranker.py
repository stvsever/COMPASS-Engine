"""
COMPASS Clinical Relevance Ranker Tool

Ranks features by clinical relevance for the target condition.
"""

import json
from typing import Dict, Any, Optional, List

from .base_tool import BaseTool


class ClinicalRelevanceRanker(BaseTool):
    """
    Ranks features by clinical relevance for prediction.
    
    Uses medical knowledge to prioritize features that are
    most informative for neuropsychiatric/neurologic disorders.
    """
    
    TOOL_NAME = "ClinicalRelevanceRanker"
    PROMPT_FILE = "clinical_ranker.txt"
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate that required inputs are present."""
        if "target_condition" not in input_data:
            return "Missing target_condition"
        
        return None
    
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the clinical ranking prompt."""
        target = input_data.get("target_condition", "neuropsychiatric")
        
        # Get features from dependency outputs or hierarchical deviation
        dep_outputs = input_data.get("dependency_outputs", {})
        hierarchical_deviation = input_data.get("hierarchical_deviation", {})
        
        features = self._collect_features(dep_outputs, hierarchical_deviation)
        
        prompt_parts = [
            f"## TARGET CONDITION: {target}",
            
            f"\n## FEATURES TO RANK",
            f"```json\n{json.dumps(features[:30], indent=2)}\n```",
            
            f"\n## CURRENT ABNORMALITY PROFILE",
            self._get_abnormality_profile(hierarchical_deviation),
            
            "\n## CLINICAL RANKING GUIDELINES",
            self._get_clinical_guidelines(target),
            
            "\n## TASK",
            f"Rank these features by clinical relevance for {target} prediction.",
            "Consider established biomarkers and clinical evidence.",
            "Note any clinically important features that are missing."
        ]
        
        return "\n".join(prompt_parts)
    
    def _collect_features(
        self,
        dep_outputs: Dict[str, Any],
        hierarchical_deviation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Collect features from various sources."""
        features = []
        
        # From feature synthesizer output
        for step_key, output in dep_outputs.items():
            if isinstance(output, dict):
                if "top_10_features" in output:
                    features.extend(output["top_10_features"])
                if "feature_rankings" in output:
                    rankings = output["feature_rankings"]
                    if "top_10_features" in rankings:
                        features.extend(rankings["top_10_features"])
        
        # From hierarchical deviation if no features yet
        if not features and hierarchical_deviation:
            features = self._extract_from_deviation(hierarchical_deviation)
        
        return features[:30]
    
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
                    lines.append(f"- {domain}: {summary.get('severity', 'UNKNOWN')}")
            return "\n".join(lines)
        
        return "Deviation structure available but not summarized"
    
    def _get_clinical_guidelines(self, target: str) -> str:
        """Get condition-specific clinical guidelines."""
        if target == "neuropsychiatric":
            return """
For neuropsychiatric conditions, prioritize:
- HIGH: Limbic volumes, prefrontal cortex, stress markers, cognitive tests
- MEDIUM: Global brain measures, inflammatory markers
- LOWER: Motor cortex, non-specific markers
"""
        else:
            return """
For neurologic conditions, prioritize:
- HIGH: Global atrophy, white matter lesions, genetic variants, CSF markers
- MEDIUM: Cognitive performance, metabolic factors
- LOWER: Mood measures, social factors
"""
