"""
COMPASS Anomaly Narrative Builder Tool

Builds narratives from hierarchical deviation maps.
"""

import json
from typing import Dict, Any, Optional, List

from .base_tool import BaseTool


class AnomalyNarrativeBuilder(BaseTool):
    """
    Builds token-efficient narratives from hierarchical deviation maps.
    
    Creates clear, interpretable summaries of abnormality patterns
    across the multimodal data hierarchy.
    """
    
    TOOL_NAME = "AnomalyNarrativeBuilder"
    PROMPT_FILE = "anomaly_narrative.txt"
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate that required inputs are present."""
        if not input_data.get("hierarchical_deviation"):
            return "Missing hierarchical_deviation data"
        
        if "target_condition" not in input_data:
            return "Missing target_condition"
        
        return None
    
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the narrative building prompt."""
        target = input_data.get("target_condition", "neuropsychiatric")
        hierarchical_deviation = input_data.get("hierarchical_deviation", {})
        
        # Analyze the deviation structure
        analysis = self._analyze_deviation(hierarchical_deviation)
        
        prompt_parts = [
            f"## TARGET CONDITION: {target}",
            
            f"\n## HIERARCHICAL DEVIATION MAP",
            f"```json\n{json.dumps(analysis, indent=2)}\n```",
            
            f"\n## FULL DEVIATION STRUCTURE (TRUNCATED)",
            str(hierarchical_deviation)[:4000],
            
            "\n## TASK",
            "Build a token-efficient narrative from this hierarchical deviation map.",
            "Identify affected domains, subsystems, and key abnormalities.",
            f"Focus on patterns relevant to {target}.",
            "Create a scannable, clinically useful narrative."
        ]
        
        return "\n".join(prompt_parts)
    
    def _analyze_deviation(self, deviation: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-analyze the deviation structure."""
        analysis = {
            "total_features": 0,
            "abnormal_features": 0,
            "domains_summary": {},
            "severity_distribution": {
                "SEVERE": 0,
                "MODERATE": 0,
                "MILD": 0,
                "NORMAL": 0
            },
            "most_extreme_features": []
        }
        
        all_features = []
        
        def traverse(node: Dict[str, Any], domain: str = "ROOT"):
            z = node.get("z_score")
            
            # Update domain on first level children
            current_domain = domain
            if domain == "ROOT" and "node_name" in node:
                current_domain = node["node_name"]
            
            if z is not None:
                analysis["total_features"] += 1
                
                severity = self._classify_severity(z)
                analysis["severity_distribution"][severity] += 1
                
                if severity != "NORMAL":
                    analysis["abnormal_features"] += 1
                    all_features.append({
                        "name": node.get("node_name", ""),
                        "z_score": z,
                        "domain": current_domain,
                        "severity": severity
                    })
                
                # Track by domain
                if current_domain not in analysis["domains_summary"]:
                    analysis["domains_summary"][current_domain] = {
                        "total": 0,
                        "abnormal": 0,
                        "direction": {"HIGH": 0, "LOW": 0}
                    }
                
                analysis["domains_summary"][current_domain]["total"] += 1
                if severity != "NORMAL":
                    analysis["domains_summary"][current_domain]["abnormal"] += 1
                
                if z > 0:
                    analysis["domains_summary"][current_domain]["direction"]["HIGH"] += 1
                else:
                    analysis["domains_summary"][current_domain]["direction"]["LOW"] += 1
            
            for child in node.get("children", []):
                traverse(child, current_domain)
        
        if "root" in deviation:
            traverse(deviation["root"])
        
        # Get most extreme features
        all_features.sort(key=lambda f: abs(f.get("z_score", 0)), reverse=True)
        analysis["most_extreme_features"] = all_features[:10]
        
        return analysis
    
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
