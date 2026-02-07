"""
COMPASS Differential Diagnosis Tool

Generates differential diagnoses based on accumulated evidence.
Scheduled late in orchestration, after hypothesis generation and clinical ranking.
"""

import json
from typing import Dict, Any, Optional, List

from .base_tool import BaseTool


class DifferentialDiagnosis(BaseTool):
    """
    Generates differential diagnoses from accumulated evidence.
    
    This tool runs LATE in the pipeline, after hypothesis generation
    and clinical ranking, to provide rule-out logic and likelihood scoring.
    """
    
    TOOL_NAME = "DifferentialDiagnosis"
    PROMPT_FILE = "differential_diagnosis.txt"
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate that required inputs are present."""
        if "target_condition" not in input_data:
            return "Missing target_condition"
        return None
    
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the differential diagnosis prompt."""
        target = input_data.get("target_condition", "neuropsychiatric")
        control = input_data.get("control_condition", "")
        hierarchical_deviation = input_data.get("hierarchical_deviation", {})
        non_numerical_data = input_data.get("non_numerical_data", "")
        dep_outputs = input_data.get("dependency_outputs", {})
        
        # Collect evidence from dependency outputs
        hypotheses = self._extract_hypotheses(dep_outputs)
        ranked_features = self._extract_ranked_features(dep_outputs)
        phenotype = self._extract_phenotype(dep_outputs)
        
        # Get abnormality summary
        abnormality_summary = self._get_abnormality_summary(hierarchical_deviation)
        
        prompt_parts = [
            f"## TARGET CONDITION: {target}",
            f"## CONTROL CONDITION: {control}",
            
            f"\n## PHENOTYPE REPRESENTATION",
            phenotype if phenotype else "Not available",
            
            f"\n## HYPOTHESES GENERATED",
            f"```json\n{json.dumps(hypotheses, indent=2)}\n```" if hypotheses else "No hypotheses available",
            
            f"\n## CLINICALLY RANKED FEATURES",
            f"```json\n{json.dumps(ranked_features[:10], indent=2)}\n```" if ranked_features else "No ranked features",
            
            f"\n## ABNORMALITY PROFILE",
            abnormality_summary,
            
            f"\n## CLINICAL NOTES",
            non_numerical_data[:1500] if non_numerical_data else "No clinical notes",
            
            "\n## TASK",
            f"Generate a comprehensive differential diagnosis for {target}.",
            "Include rule-out criteria and likelihood scores for each diagnosis.",
            "Consider both common and rare conditions that fit the phenotype."
        ]
        
        return "\n".join(prompt_parts)
    
    def _extract_hypotheses(self, dep_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract hypotheses from dependency outputs."""
        hypotheses = []
        
        for step_key, output in dep_outputs.items():
            if isinstance(output, dict):
                if "primary_hypothesis" in output:
                    hypotheses.append(output["primary_hypothesis"])
                if "alternative_hypotheses" in output:
                    hypotheses.extend(output["alternative_hypotheses"])
        
        return hypotheses[:5]
    
    def _extract_ranked_features(self, dep_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract ranked features from dependency outputs."""
        features = []
        
        for step_key, output in dep_outputs.items():
            if isinstance(output, dict):
                if "ranked_features" in output:
                    features.extend(output["ranked_features"])
                if "top_clinical_priorities" in output:
                    for priority in output["top_clinical_priorities"]:
                        features.append({"feature": priority, "rank": len(features) + 1})
        
        return features[:15]
    
    def _extract_phenotype(self, dep_outputs: Dict[str, Any]) -> str:
        """Extract phenotype representation from dependency outputs."""
        for step_key, output in dep_outputs.items():
            if isinstance(output, dict):
                if "phenotype_summary" in output:
                    return output["phenotype_summary"]
                if "clinical_phenotype" in output:
                    return json.dumps(output["clinical_phenotype"], indent=2)
        return ""
    
    def _get_abnormality_summary(self, deviation: Dict[str, Any]) -> str:
        """Summarize abnormalities from deviation structure."""
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
        
        return "Deviation data available but not summarized"

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
