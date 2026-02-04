"""
COMPASS Phenotype Representation Tool

Generates comprehensive phenotype representation including subtypes and heterogeneity.
Scheduled early in orchestration to inform downstream tools.
"""

import json
from typing import Dict, Any, Optional, List

from .base_tool import BaseTool


class PhenotypeRepresentation(BaseTool):
    """
    Generates detailed phenotype representation for the participant.
    
    This tool runs EARLY in the pipeline to create a comprehensive
    clinical picture that informs all downstream analysis.
    """
    
    TOOL_NAME = "PhenotypeRepresentation"
    PROMPT_FILE = "phenotype_representation.txt"
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate that required inputs are present."""
        if "target_condition" not in input_data:
            return "Missing target_condition"
        return None
    
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the phenotype representation prompt."""
        target = input_data.get("target_condition", "neuropsychiatric")
        hierarchical_deviation = input_data.get("hierarchical_deviation", {})
        non_numerical_data = input_data.get("non_numerical_data", "")
        domain_data = input_data.get("domain_data", {})
        
        prompt_parts = [
            f"## TARGET CONDITION: {target}",
            
            "\n## TASK",
            f"Generate a comprehensive, general 'Gold Standard' phenotype definition for {target}.",
            "Do NOT reference any specific patient data. This is a conceptual definition step.",
            "Define:",
            "1. Clinical Presentation & Core (and detailled) Symptoms",
            "2. Potential Subtypes",
            "3. Biomarker profiles (what would you EXPECT to see in a case?)",
            "4. Heterogeneity patterns",
            
            "This representation will serve as the template against which patient data is compared later."
        ]
        
        return "\n".join(prompt_parts)
    
    def _extract_clinical_features(self, deviation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract significant clinical features from deviation structure."""
        features = []
        
        def traverse(node: Dict[str, Any], path: str = ""):
            z_score = node.get("z_score")
            if z_score is not None and abs(z_score) > 1.0:
                features.append({
                    "feature": node.get("node_name", ""),
                    "path": path,
                    "z_score": z_score,
                    "direction": node.get("direction"),
                    "severity": node.get("severity")
                })
            
            for child in node.get("children", []):
                child_path = f"{path}/{child.get('node_name', '')}" if path else child.get("node_name", "")
                traverse(child, child_path)
        
        if "root" in deviation:
            traverse(deviation["root"])
        
        # Sort by absolute z-score
        features.sort(key=lambda f: abs(f.get("z_score", 0)), reverse=True)
        return features
    
    def _extract_demographics(self, non_numerical: str) -> str:
        """Extract demographics from non-numerical text."""
        if not non_numerical:
            return ""
        
        # Return first portion for context
        return non_numerical[:2000]
    
    def _summarize_domains(self, domain_data: Dict[str, Any]) -> str:
        """Summarize available domain data."""
        if not domain_data:
            return "No domain-specific data provided"
        
        lines = []
        for domain, data in domain_data.items():
            if isinstance(data, dict):
                node_count = self._count_nodes(data)
                lines.append(f"- {domain}: {node_count} data points")
            else:
                lines.append(f"- {domain}: available")
        
        return "\n".join(lines) if lines else "No domains"
    
    def _count_nodes(self, data: Any, count: int = 0) -> int:
        """Count nodes in nested structure."""
        if isinstance(data, dict):
            count += 1
            for v in data.values():
                count = self._count_nodes(v, count)
        elif isinstance(data, list):
            for item in data:
                count = self._count_nodes(item, count)
        return count
