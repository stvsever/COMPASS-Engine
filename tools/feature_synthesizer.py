"""
COMPASS Feature Synthesizer Tool

Synthesizes feature importance from hierarchical data structures.
"""

import json
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
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate that required inputs are present."""
        if not input_data.get("hierarchical_deviation"):
            return "Missing hierarchical_deviation data"
        
        if "target_condition" not in input_data:
            return "Missing target_condition"
        
        return None
    
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the feature synthesis prompt."""
        target = input_data.get("target_condition", "neuropsychiatric")
        hierarchical_deviation = input_data.get("hierarchical_deviation", {})
        domains = input_data.get("input_domains", [])
        
        # Extract features with z-scores
        features = self._extract_features(hierarchical_deviation)
        
        prompt_parts = [
            f"## TARGET CONDITION: {target}",
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
            "Group by domain and provide aggregate assessments."
        ]
        
        return "\n".join(prompt_parts)
    
    def _extract_features(self, deviation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all features with their z-scores."""
        features = []
        
        def traverse(node: Dict[str, Any], path: List[str]):
            current_path = path + [node.get("node_name", node.get("name", "unknown"))]
            
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
            
            # Traverse children (excluding _stats)
            for key, child in node.items():
                if key != "_stats" and isinstance(child, dict):
                    # Add name to child if missing (using key)
                    if "node_name" not in child:
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
